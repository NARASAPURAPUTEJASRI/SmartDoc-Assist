from gtts import gTTS
import os

# Example text in Hindi
text = "नमस्ते, आप कैसे हैं?"

# Choose language (hi = Hindi, te = Telugu, ta = Tamil, etc.)
tts = gTTS(text=text, lang='te')

# Save to file
tts.save("output.mp3")

# Play the audio (Windows)
os.system("start output.mp3")
