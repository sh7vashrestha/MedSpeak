
import argparse
import whisper

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--asr_model", default="small")
    args = ap.parse_args()

    model = whisper.load_model(args.asr_model)
    res = model.transcribe(args.audio, language="en")
    print(res["text"].strip())

if __name__ == "__main__":
    main()
