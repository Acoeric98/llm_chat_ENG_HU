from gpt4all import GPT4All
from pathlib import Path
from datetime import datetime
import secrets
import time
import sys

# --- Models & runtime config ---
model_name2 = "SambaLingo-Hungarian-Chat-Q5_K_S.gguf"   # Hungarian
model_name  = "Meta-Llama-3-8B-Instruct.Q4_0.gguf"      # English
custom_dir  = Path(r"F:\AI_Models\gpt4all")
device_type = "cuda"

# --- Logging setup ---
logs_dir = Path("chat_logs")
logs_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
rand_id = secrets.token_hex(3)
log_path = logs_dir / f"chat_{timestamp}_{rand_id}.pychat"

def make_stream_printer(file_handle):
    prev = {"text": ""}

    def _printer(token_id, chunk):
        try:
            if chunk.startswith(prev["text"]):
                new_part = chunk[len(prev["text"]):]
                if new_part:
                    sys.stdout.write(new_part)
                    sys.stdout.flush()
                    file_handle.write(new_part)
                    file_handle.flush()
                prev["text"] = chunk
            else:
                sys.stdout.write(chunk)
                sys.stdout.flush()
                file_handle.write(chunk)
                file_handle.flush()
                prev["text"] += chunk
        except Exception:
            safe = chunk.encode("utf-8", "ignore").decode("utf-8", "ignore")
            if safe.startswith(prev["text"]):
                new_part = safe[len(prev["text"]):]
                if new_part:
                    sys.stdout.write(new_part)
                    sys.stdout.flush()
                    file_handle.write(new_part)
                    file_handle.flush()
                prev["text"] = safe
            else:
                sys.stdout.write(safe)
                sys.stdout.flush()
                file_handle.write(safe)
                file_handle.flush()
                prev["text"] += safe
        return True

    return _printer


def pick_language_startup():
    while True:
        sel = input("Select language [en=English, hu=Magyar]: ").strip().lower()
        if sel in ("en", "hu"):
            return sel
        print("Please type 'en' or 'hu'.")


def build_llm(lang_code: str) -> tuple[GPT4All, str]:
    """Create an LLM instance based on the language code."""
    selected = model_name if lang_code == "en" else model_name2
    print("MODEL LOADED:", selected)
    llm = GPT4All(
        selected,
        model_path=custom_dir,
        allow_download=False,
        device=device_type
    )
    return llm, selected


def run_chat_session(llm: GPT4All, model_id: str, lang_code: str, log_file) -> tuple[str, str]:
    """
    Run a chat session loop with the given model.
    Returns a tuple: (action, payload)
      - ("exit", "") to terminate program
      - ("switch", "en"/"hu") to switch language/model
    """
    # One system prompt for both; concise answers + memory across turns.
    system_prompt = "You are a helpful assistant. Keep answers concise and remember prior turns."

    # Write session header to log
    session_start = datetime.now().isoformat(timespec="seconds")
    log_file.write(f"# Chat session started: {session_start}\n")
    log_file.write(f"# Model: {model_id}\n")
    log_file.write(f"# Device: {device_type}\n")
    log_file.write("# ----------------------------------------\n\n")
    log_file.flush()

    welcome = "Welcome! How can I help you today?"
    with llm.chat_session(system_prompt=system_prompt):
        print(welcome)
        print("Type '/lang en' or '/lang hu' to switch model. Type 'exit' to quit.")
        log_file.write(f"SYSTEM: {welcome}\n\n")
        log_file.flush()

        while True:
            print("x" * 90)
            prompt = input("You: ")
            if prompt.lower() == "exit":
                end_ts = datetime.now().isoformat(timespec="seconds")
                print("Goodbye.")
                log_file.write("\n# ----------------------------------------\n")
                log_file.write(f"# Chat session ended: {end_ts}\n")
                log_file.flush()
                return "exit", ""

            # Handle language switch command
            if prompt.lower().startswith("/lang"):
                parts = prompt.split()
                if len(parts) == 2 and parts[1].lower() in ("en", "hu"):
                    new_lang = parts[1].lower()
                    switch_ts = datetime.now().isoformat(timespec="seconds")
                    log_file.write(f"\n# ---- Language switch requested: {new_lang.upper()} at {switch_ts} ----\n\n")
                    log_file.flush()
                    print(f"Switching language to {new_lang.upper()}...")
                    return "switch", new_lang
                else:
                    print("Usage: /lang en   or   /lang hu")
                    continue

            # Normal turn
            turn_ts = datetime.now().isoformat(timespec="seconds")
            log_file.write(f"[{turn_ts}] USER: {prompt}\n")
            log_file.write("AI: ")
            log_file.flush()

            start_time = time.time()
            print("\nAI: ", end="", flush=True)
            printer = make_stream_printer(log_file)

            _ = llm.generate(
                prompt,
                max_tokens=8192,
                temp=0.8,
                top_p=0.9,
                top_k=50,
                repeat_penalty=1.1,
                callback=printer
            )

            end_time = time.time()
            delta = end_time - start_time

            print("\n")
            print(f"Current device: {device_type}")
            print(f"Elapsed time: {delta:.2f} seconds")

            log_file.write("\n\n")
            log_file.write(f"[META] device={device_type} elapsed_seconds={delta:.2f}\n\n")
            log_file.flush()


def main():
    # Pick initial language
    current_lang = pick_language_startup()

    with open(log_path, "w", encoding="utf-8") as log_file:
        while True:
            llm, model_id = build_llm(current_lang)
            action, payload = run_chat_session(llm, model_id, current_lang, log_file)

            if action == "exit":
                break
            elif action == "switch":
                current_lang = payload
                continue

    print(f"Session log saved to: {log_path.resolve()}")


if __name__ == "__main__":
    main()
