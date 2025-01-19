from groq import Groq
import base64


# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
import cv2

# Open the webcam with optimizations
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow for faster access on Windows (skip on Linux/Mac)

# Check if the webcam is accessible
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# Capture a single frame immediately
ret, frame = cap.read()

if ret:
    # Save the image to a file
    file_name = "captured_image.jpg"
    cv2.imwrite(file_name, frame)
    print(f"Image saved as {file_name}")
else:
    print("Error: Failed to capture image.")

# Release the webcam
cap.release()


# Path to your image
image_path = file_name

# Getting the base64 string
base64_image = encode_image(image_path)

client = Groq()

prompt = '''
You are an advanced language model tasked with analyzing the sentiment of an audience during a presentation. Your analysis should evaluate the following three metrics:
Audience must be looking at the camera! Not looking at the camera means they are unattentive and need a low score. Looking at the camera is a good sign of attention.
1. **Engagement**: How attentive and focused the audience appears.
2. **Mood**: The overall emotional state of the audience.
3. **Excitement**: The energy and enthusiasm level of the audience.

Use the following JSON grammar strictly to format your response:

root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4}) # escapes
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [0-9] [1-9]{0,15})? ws

# Optional space: by convention, applied in this grammar after literal chars when allowed
ws ::= | " " | "\n" [ \t]{0,20}

### Output Requirements:
- Return only a single JSON object adhering to this structure.
- The JSON object must include the following keys:
  - `"engagement"`: A number between 0 and 10.
  - `"mood"`: A number between 0 and 10.
  - `"excitement"`: A number between 0 and 10.

### Example Output:
```json
{
  "engagement": 8,
  "mood": 7,
  "excitement": 9
}

'''

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        }
    ],
    model="llama-3.2-90b-vision-preview",
)

print(chat_completion.choices[0].message.content)