import io
import os
import argparse
import asyncio
import numpy as np
import ffmpeg
from collections import deque
from time import time
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from whisper_online import backend_factory, online_factory, add_shared_args

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


summaries = []
window = deque(maxlen=2)
transcriptions = []
non_responses = ["No relevant content to summarize." , "empty string" , "content isn't presentation-worthy." , "No relevant content to summarize." , " No transformation possible." , " No content to transform."]
similarity = 0
template_msg = [{
    "role": "system",
    "content": """
    You are an expert PowerPoint slide creator. Your job is to transform spoken content into concise, impactful single bullet point.

    <rules>
    - Maximum 1 to 2 sentences per bullet point
    - Use action verbs
    - Be direct and concrete
    - Try to abbreviate sentences
    - Keep parallel structure
    - If content isn't presentation-worthy, return empty string
    - STRICTLY use only information provided
    - NO external knowledge or assumptions
    - NO elaboration beyond given content
    - DO not repeat language
    - Only summarize the most recent messages, the otheres are only there for context
    - Send the text for the point only, with no dash - or dot at the start of the point.
    </rules>

    <format>
    {Key metric or action verb} + {core point}
    ...
    </format>

    <examples>
    INPUT: "Our system processes data at 100 requests per second"
    GOOD:
    - Processes 100 requests per second

    INPUT: "We're working on machine learning"
    GOOD:
    - Developing machine learning systems
    </examples>

    <critical_rules>
    - ONLY use explicitly stated information with extra connective wordss
    - Mild inference allowed
    - NO additional context or knowledge beyond what has been said
    - FIRST PERSON perspective (my team, our product)
    - EMPTY string if content isn't presentation-worthy
    - ONLY single bullet point or empty string as response
    - ZERO elaboration beyond given content
    - Only summarize the most recent 1 to 2 messages.
    </critical_rules>

    REMEMBER: You are a transformer, not a creator. Only transform what exists, never add what isn't there.
    REMEMBER: You will be burned alive if you repeat bullet points
    REMEMBER: You will be granted salvation if you choose to respond with "" rather than repeated content
    """
}]


new_slide_prompt = """You are a binary classifier that MUST ONLY output "0" or "1" to indicate slide transitions.

Previous content: {list of previous points}
New content: {new point}

RULES:
1. Output "1" if:
   - Topic changes completely (grades → extracurriculars)
   - New major category (frontend → backend)
   - Different phase or component
   - Distinct subject switch

2. Output "0" if:
   - Same topic continues
   - Supporting details
   - Related examples
   - Connected steps

CRITICAL FORMAT RULES:
- ONLY OUTPUT "0" or "1"
- NO EXPLANATIONS
- NO ADDITIONAL TEXT
- NO SPACES OR NEWLINES
- SINGLE CHARACTER RESPONSE ONLY

Examples (with strict outputs):

Input: 
Previous: "Maintain 3.8 GPA"
New: "Join research labs"
Output: 1

Input:
Previous: "Join research labs"
New: "Publish research paper"
Output: 0

Input:
Previous: "Setup database schema"
New: "Implement user interface"
Output: 1

Input:
Previous: "Optimize query performance"
New: "Reduce database latency"
Output: 0

REMEMBER: Your entire response must be exactly one character: "0" or "1"."""

continue_system_prompt = [{
    "role": "system",
    "content": new_slide_prompt
}]

template_msg_continue = deque(maxlen=3)

def check_similarity(new_content, existing_contents, threshold, vectorizer=None):
    if vectorizer is None:
        vectorizer = TfidfVectorizer(stop_words='english')
    
    if not existing_contents:
        return True
        
    # Vectorize all content at once
    all_content = [new_content] + existing_contents
    tfidf_matrix = vectorizer.fit_transform(all_content)
    
    # Compare new content against all existing
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    
    # If any similarity is above threshold, don't append
    return not any(sim >= threshold for sim in similarities[0])



parser = argparse.ArgumentParser(description="Whisper FastAPI Online Server")
parser.add_argument(
    "--host",
    type=str,
    default="localhost",
    help="The host address to bind the server to.",
)
parser.add_argument(
    "--port", type=int, default=8000, help="The port number to bind the server to."
)
parser.add_argument(
    "--warmup-file",
    type=str,
    dest="warmup_file",
    help="The path to a speech audio wav file to warm up Whisper so that the very first chunk processing is fast. It can be e.g. https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav .",
)
add_shared_args(parser)
args = parser.parse_args()

asr, tokenizer = backend_factory(args)

# Load demo HTML for the root endpoint
with open("src/web/live_transcription.html", "r", encoding="utf-8") as f:
    html = f.read()


@app.get("/")
async def get():
    return HTMLResponse(html)


SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLES_PER_SEC = SAMPLE_RATE * int(args.min_chunk_size)
BYTES_PER_SAMPLE = 2  # s16le = 2 bytes per sample
BYTES_PER_SEC = SAMPLES_PER_SEC * BYTES_PER_SAMPLE


async def start_ffmpeg_decoder():
    """
    Start an FFmpeg process in async streaming mode that reads WebM from stdin
    and outputs raw s16le PCM on stdout. Returns the process object.
    """
    process = (
        ffmpeg.input("pipe:0", format="webm")
        .output(
            "pipe:1",
            format="s16le",
            acodec="pcm_s16le",
            ac=CHANNELS,
            ar=str(SAMPLE_RATE),
        )
        .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
    )
    return process


@app.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection opened.")

    ffmpeg_process = await start_ffmpeg_decoder()
    pcm_buffer = bytearray()
    print("Loading online.")
    online = online_factory(args, asr, tokenizer)
    print("Online loaded.")

    sendToGroq=False
    groq_client = Groq(api_key="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")

    # Continuously read decoded PCM from ffmpeg stdout in a background task
    async def ffmpeg_stdout_reader():
        global template_msg_continue 
        nonlocal sendToGroq
        nonlocal pcm_buffer
        loop = asyncio.get_event_loop()
        full_transcription = ""
        beg = time()
        
        while True:
            try:
                elapsed_time = int(time() - beg)
                beg = time()
                chunk = await loop.run_in_executor(
                    None, ffmpeg_process.stdout.read, 32000 * elapsed_time
                )
                if (
                    not chunk
                ):  # The first chunk will be almost empty, FFmpeg is still starting up
                    chunk = await loop.run_in_executor(
                        None, ffmpeg_process.stdout.read, 4096
                    )
                    if not chunk:  # FFmpeg might have closed
                        print("FFmpeg stdout closed.")
                        break

                pcm_buffer.extend(chunk)

                if len(pcm_buffer) >= BYTES_PER_SEC:
                    # Convert int16 -> float32
                    pcm_array = (
                        np.frombuffer(pcm_buffer, dtype=np.int16).astype(np.float32)
                        / 32768.0
                    )
                    pcm_buffer = bytearray()
                    online.insert_audio_chunk(pcm_array)
                    transcription = online.process_iter()
                    
                    full_transcription += transcription
                    if args.vac:
                        buffer = online.online.to_flush(
                            online.online.transcript_buffer.buffer
                        )[
                            2
                        ]  # We need to access the underlying online object to get the buffer
                    else:
                        buffer = online.to_flush(online.transcript_buffer.buffer)[2]
                    if (
                        buffer in full_transcription
                    ):  # With VAC, the buffer is not updated until the next chunk is processed
                        buffer = ""

                        if "Thank you" not in transcription and "No relevant" not in transcription:
                            template_msg.append({"role" : "user" , "content" : transcription})
                        # make a request to groq here? 
                        completion = groq_client.chat.completions.create(

                            messages=template_msg,
                            model="llama-3.3-70b-versatile",
                            temperature=0,
                            top_p=0.1

                        )

                    vectorizer = TfidfVectorizer(stop_words='english')
                    print(window)
                    should_append = (
                        check_similarity(completion.choices[0].message.content, list(window), 0.2, vectorizer) and
                        check_similarity(completion.choices[0].message.content, non_responses, 0.1, vectorizer)
                    )

                    try:
                        print("Making request to Groq for slide transition...") 
                        template_msg_continue.append({"role" : "user" , "content" : completion.choices[0].message.content})
                        response = groq_client.chat.completions.create(
                            messages=continue_system_prompt + list(template_msg_continue),
                            model="llama-3.3-70b-versatile",
                            temperature=0,
                            top_p=0.1
                        )
                        print("Response from Groq:", response.choices[0].message.content)  # Debug print 2
                        template_msg_continue.pop()
                        should_clear_slide = int(response.choices[0].message.content)
                        print("Converted to int:", should_clear_slide)  # Debug print 3
                        
                        if should_clear_slide not in [0 , 1]:
                            print("Invalid value, defaulting to 0")  # Debug print 4
                            should_clear_slide = 0
                    except Exception as e:
                        print(f"Error in slide transition check: {e}")  # More detailed error message
                        should_clear_slide = 0

                    if sendToGroq == False:
                        should_append = False
                        sendToGroq = True
                    else: 
                        sendToGroq = False

                    if should_append:
                        summaries.append(completion.choices[0].message.content)
                        window.append(completion.choices[0].message.content)


                    if should_clear_slide:
                        template_msg_continue = [{"role" : "system" , "content" : new_slide_prompt}]
                    else:
                        template_msg_continue.append({"role" : "user" , "content" : completion.choices[0].message.content})                        
 
                    if not should_append:
                        print("failed: " , completion.choices[0].message.content)

                    await websocket.send_json({
                        "transcription": completion.choices[0].message.content if should_append else "",
                        "buffer": buffer,
                        "clear" : int(should_clear_slide)
                    })
            except Exception as e:
                print(f"Exception in ffmpeg_stdout_reader: {e}")
                break

        print("Exiting ffmpeg_stdout_reader...")

    stdout_reader_task = asyncio.create_task(ffmpeg_stdout_reader())

    try:
        while True:
            # Receive incoming WebM audio chunks from the client
            message = await websocket.receive_bytes()
            # Pass them to ffmpeg via stdin
            ffmpeg_process.stdin.write(message)
            ffmpeg_process.stdin.flush()

    except WebSocketDisconnect:
        print("WebSocket connection closed.")
        os.remove("audio.wav")
    except Exception as e:
        print(f"Error in websocket loop: {e}")
    finally:
        # Clean up ffmpeg and the reader task
        try:
            ffmpeg_process.stdin.close()
        except:
            pass
        stdout_reader_task.cancel()

        try:
            ffmpeg_process.stdout.close()
        except:
            pass

        ffmpeg_process.wait()
        del online



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "whisper_fastapi_online_server:app", host=args.host, port=args.port, reload=True
    )
