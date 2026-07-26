"""
main.py - Run the full pipeline with one command

Usage:
    py -3.11 main.py                  <- full pipeline
    py -3.11 main.py --skip-clean     <- skip cleaning (data already cleaned)
    py -3.11 main.py --skip-extract   <- skip extraction (CSVs already exist)
    py -3.11 main.py --audio-only     <- train audio model only
    py -3.11 main.py --image-only     <- train image model only
"""
import sys
import argparse
import subprocess
from pathlib import Path

SCRIPTS = Path(__file__).parent / "scripts"


def log(msg):
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def run(script: str, *extra_args) -> bool:
    cmd = [sys.executable, str(SCRIPTS / script)] + list(extra_args)
    log(f"  >> {script}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        log(f"  [ERROR] {script} failed with exit code {result.returncode}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-clean",   action="store_true",
                        help="Skip data cleaning step")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip feature extraction step")
    parser.add_argument("--audio-only",   action="store_true",
                        help="Train audio model only")
    parser.add_argument("--image-only",   action="store_true",
                        help="Train image model only")
    args = parser.parse_args()

    log("=" * 55)
    log("  Multimodal Depression Detection System")
    log("  600 audio + 2000 images | 80% train / 20% test")
    log("=" * 55)

    if not args.skip_clean:
        log("\n--- STEP 1: DATA CLEANING + 80/20 SPLIT ---")
        if not run("clean_data.py"):
            return

    if not args.skip_extract:
        log("\n--- STEP 2a: AUDIO FEATURE EXTRACTION ---")
        if not run("extract_audio_features.py"):
            return

        log("\n--- STEP 2b: IMAGE FEATURE EXTRACTION ---")
        if not run("extract_image_features.py"):
            return

    log("\n--- STEP 3: TRAINING + EVALUATION ---")
    train_args = []
    if args.audio_only: train_args.append("--audio")
    if args.image_only: train_args.append("--image")
    if not run("train_model.py", *train_args):
        return

    log("\n" + "=" * 55)
    log("  PIPELINE COMPLETE!")
    log("  Check results/ folder for accuracy plots")
    log("  Now start the UI:  py -3.11 app.py")
    log("  Then open:         http://127.0.0.1:5000")
    log("=" * 55)


if __name__ == "__main__":
    main()