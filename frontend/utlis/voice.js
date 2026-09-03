export function speak(text) {
  if ('speechSynthesis' in window) {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "en-US"; // you can set "hi-IN", "te-IN", etc. for Indian voices
    speechSynthesis.cancel(); // stop previous speech
    speechSynthesis.speak(utterance);
  } else {
    console.warn("Speech Synthesis not supported in this browser.");
  }
}
