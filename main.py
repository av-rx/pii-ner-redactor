import logging
import os
import re
import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Cache lives next to this file, not the current working directory, so running from
# another directory does not trigger a second 1.3 GB download.
LOCAL_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_cache")
os.makedirs(LOCAL_CACHE, exist_ok=True)
os.environ["HF_HOME"] = LOCAL_CACHE
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"          # suppress TF C++ / oneDNN info logs
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # suppress Windows symlinks warning

log = logging.getLogger(__name__)

EMAIL_PATTERN = r'\b[\w.+%-]+@[\w.-]+\.\w+\b'
SSN_PATTERN = r'\b\d{3}-\d{2}-\d{4}\b'
# 4-4-4-4 (Visa/MC style), 4-6-5 (Amex style), or 13-16 contiguous digits
CREDIT_CARD_PATTERN = (
    r'\b(?:\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}'
    r'|\d{4}[ -]?\d{6}[ -]?\d{5}'
    r'|\d{13,16})\b'
)
# house number, up to five words, a street-type word (whole word), optional apt/unit.
# No newline anywhere so a match can never bridge two lines.
_STREET_TYPES = (
    r'Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|Lane|Ln|Drive|Dr|Court|Ct|'
    r'Circle|Cir|Way|Wy|Square|Sq|Place|Pl|Terrace|Ter'
)
ADDRESS_PATTERN = (
    r'\b\d{1,5}[ \t]+'
    r"(?:[A-Za-z0-9.'-]+[ \t]+){0,5}?"
    rf'(?:{_STREET_TYPES})\b\.?'
    r'(?:,?[ \t]*(?:Apt|Unit|Suite|#)[ \t]*\w+)?'
)
# optional +country, optional (area), then 2-4 groups of 2-4 digits with one separator
# each. Digit count is checked in _phone_repl (7..16) so "1500.00" is left alone.
# The lookahead rejects ISO dates; the lookbehind stops matches starting mid-number.
PHONE_PATTERN = (
    r'(?<![\w-])(?!\d{4}-\d{2}-\d{2}\b)'
    r'(?:\+\d{1,3}[-.\s]?)?'
    r'(?:\(\d{2,4}\)[-.\s]?)?'
    r'(?:\d{2,4}(?:[-.\s]\d{2,4}){1,3}|\d{10,11})\b'
)
PHONE_MIN_DIGITS, PHONE_MAX_DIGITS = 7, 16
# one or two letters then 6-8 digits; covers the common "A1234567" family
PASSPORT_PATTERN = r'\b[A-Z]{1,2}\d{6,8}\b'
# NER runs first and tags "Maple Boulevard" as LOC, leaving "2640 [REDACTED_LOC]". A short
# number directly before a location placeholder is a house number, so fold it in.
HOUSE_NUMBER_BEFORE_LOC_PATTERN = r'\b\d{1,5}[ \t]+\[REDACTED_LOC\]'


def _phone_repl(m):
    digits = sum(ch.isdigit() for ch in m.group())
    if PHONE_MIN_DIGITS <= digits <= PHONE_MAX_DIGITS:
        return "[REDACTED_PHONE]"
    return m.group()


# Order matters: each pattern runs over the output of the previous one, so the more
# specific shapes (SSN, card) go before the permissive phone pattern.
PATTERNS = [
    (EMAIL_PATTERN, "[REDACTED_EMAIL]"),
    (SSN_PATTERN, "[REDACTED_SSN]"),
    (CREDIT_CARD_PATTERN, "[REDACTED_CREDIT_CARD]"),
    (ADDRESS_PATTERN, "[REDACTED_ADDRESS]"),
    (HOUSE_NUMBER_BEFORE_LOC_PATTERN, "[REDACTED_ADDRESS]"),
    (PHONE_PATTERN, _phone_repl),
    (PASSPORT_PATTERN, "[REDACTED_PASSPORT]"),
]

# BERT's position embeddings stop at 512 tokens; the pipeline silently truncates beyond
# that, so NER input is split into chunks that stay under this budget.
NER_MAX_TOKENS = 400

ner_pipeline = None
loader_queue = queue.Queue()
last_ml_error = None  # set by redact_text when the model raised and regex-only output was returned


def load_ner_model_in_background(model_name="dbmdz/bert-large-cased-finetuned-conll03-english"):
    global ner_pipeline
    try:
        loader_queue.put(("status", "Importing transformers..."))
        # Import inside thread to avoid blocking main thread at startup
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        from transformers import pipeline
        import transformers
        transformers.logging.set_verbosity_error()  # suppress "some weights not used" etc.

        loader_queue.put(("status", f"Loading model: {model_name}"))
        ner_pipeline = pipeline("ner", model=model_name, tokenizer=model_name, aggregation_strategy="simple", device=-1)  # -1 = CPU
        loader_queue.put(("done", "Model loaded."))
    except Exception as exc:
        ner_pipeline = None
        loader_queue.put(("error", f"Model failed to load: {exc}"))


def chunk_text(text, max_tokens, count_tokens):
    """Split text into pieces that each fit under max_tokens and concatenate back to text.

    Splits on line boundaries first; a single line over budget is split on whitespace.
    """
    chunks = []
    for line in text.splitlines(keepends=True):
        if count_tokens(line) <= max_tokens:
            chunks.append(line)
            continue
        current = ""
        for piece in re.split(r'(\s+)', line):
            if not piece:
                continue
            candidate = current + piece
            if current and not piece.isspace() and count_tokens(candidate) > max_tokens:
                chunks.append(current)
                current = piece
            else:
                current = candidate
        if current:
            chunks.append(current)
    return chunks


def _make_token_counter(pipe):
    tokenizer = getattr(pipe, "tokenizer", None)
    if tokenizer is None:
        return lambda s: len(s.split())
    return lambda s: len(tokenizer(s)["input_ids"])


def _redact_ner_chunk(chunk, pipe):
    body = chunk.rstrip("\n")
    tail = chunk[len(body):]
    if not body.strip():
        return chunk
    entities = sorted(pipe(body), key=lambda e: e.get("start", 0))
    redacted = body
    offset = 0
    last_end = -1
    for ent in entities:
        start, end = ent.get("start"), ent.get("end")
        if start is None or end is None or start < last_end:
            continue  # malformed or overlapping span: skip rather than corrupt the string
        last_end = end
        redaction = f"[REDACTED_{ent.get('entity_group', 'ENTITY')}]"
        redacted = redacted[:start + offset] + redaction + redacted[end + offset:]
        offset += len(redaction) - (end - start)
    return redacted + tail


def redact_text(text: str) -> str:
    global last_ml_error
    last_ml_error = None
    redacted_text = text

    pipe = ner_pipeline  # snapshot: the loader thread may assign the global mid-call
    if pipe:
        try:
            chunks = chunk_text(text, NER_MAX_TOKENS, _make_token_counter(pipe))
            redacted_text = "".join(_redact_ner_chunk(c, pipe) for c in chunks)
        except Exception as exc:
            last_ml_error = f"{type(exc).__name__}: {exc}"
            log.warning("NER failed, returning regex-only output: %s", last_ml_error)
            redacted_text = text

    # Regex runs after ML, catching structured PII the model misses
    for pat, repl in PATTERNS:
        redacted_text = re.sub(pat, repl, redacted_text, flags=re.IGNORECASE)

    return redacted_text


class PiiRedactorGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PII NER Redactor")
        self.geometry("1100x650")
        self.minsize(900, 500)

        self.result_queue = queue.Queue()
        self.model_status = "Loading..."
        self.redacting = False

        self._create_widgets()
        self._start_model_loader()  # start background loading immediately
        self.after(200, self._poll_queues)  # poll for model load and redaction results

    def _create_widgets(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        # Paned window for side-by-side text panels
        self.paned = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True)

        # Left: original
        frame_left = ttk.Labelframe(self.paned, text="Original Text")
        frame_left.pack(fill=tk.BOTH, expand=True)
        self.text_original = tk.Text(frame_left, wrap=tk.WORD)
        self.text_original.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        scroll_l = ttk.Scrollbar(frame_left, orient=tk.VERTICAL, command=self.text_original.yview)
        self.text_original.configure(yscrollcommand=scroll_l.set)
        scroll_l.pack(side=tk.RIGHT, fill=tk.Y)
        self.paned.add(frame_left, weight=1)

        # Right: redacted
        frame_right = ttk.Labelframe(self.paned, text="Redacted Text")
        frame_right.pack(fill=tk.BOTH, expand=True)
        self.text_redacted = tk.Text(frame_right, wrap=tk.WORD, fg="darkred")
        self.text_redacted.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        scroll_r = ttk.Scrollbar(frame_right, orient=tk.VERTICAL, command=self.text_redacted.yview)
        self.text_redacted.configure(yscrollcommand=scroll_r.set)
        scroll_r.pack(side=tk.RIGHT, fill=tk.Y)
        self.paned.add(frame_right, weight=1)

        # Bottom controls
        controls = ttk.Frame(self)
        controls.pack(fill=tk.X, padx=8, pady=8)

        self.load_btn = ttk.Button(controls, text="Load .TXT", command=self._on_load_file)
        self.load_btn.pack(side=tk.LEFT, padx=6)

        self.redact_btn = ttk.Button(controls, text="Redact", command=self._on_redact)
        self.redact_btn.pack(side=tk.LEFT, padx=6)

        self.save_btn = ttk.Button(controls, text="Save Redacted", command=self._on_save_file)
        self.save_btn.pack(side=tk.LEFT, padx=6)

        # Model status indicator
        self.status_var = tk.StringVar(value="Model: Loading...")
        self.status_label = ttk.Label(controls, textvariable=self.status_var)
        self.status_label.pack(side=tk.RIGHT, padx=8)

    def _set_model_status(self, text):
        self.model_status = text
        if not self.redacting:
            self.status_var.set(f"Model: {text}")

    def _start_model_loader(self):
        threading.Thread(target=load_ner_model_in_background, daemon=True).start()
        self._show_loading_dialog()

    def _show_loading_dialog(self):
        self.loading_win = tk.Toplevel(self)
        self.loading_win.title("Loading NER Model")
        self.loading_win.geometry("360x130")
        self.loading_win.transient(self)
        # non-modal so the main GUI stays responsive while the model downloads
        self.loading_win.attributes("-topmost", True)

        lbl = ttk.Label(
            self.loading_win,
            text="Model is loading in background.\nYou can use the app; ML will activate when ready.",
            wraplength=320,
        )
        lbl.pack(padx=12, pady=(12, 6))

        self.loading_status_var = tk.StringVar(value="Starting...")
        status_lbl = ttk.Label(self.loading_win, textvariable=self.loading_status_var, foreground="blue")
        status_lbl.pack(padx=12, pady=6)

        ttk.Button(self.loading_win, text="Hide", command=self.loading_win.withdraw).pack(pady=(0, 8))

    def _close_loading_dialog(self):
        try:
            self.loading_win.after(600, self.loading_win.destroy)
        except Exception:
            pass

    def _poll_queues(self):
        try:
            while True:
                kind, msg = loader_queue.get_nowait()
                if kind == "status":
                    self.loading_status_var.set(msg)
                    self._set_model_status(msg)
                elif kind == "done":
                    self.loading_status_var.set("Model ready.")
                    self._set_model_status("Ready")
                    self._close_loading_dialog()
                elif kind == "error":
                    self.loading_status_var.set("Model failed to load.")
                    self._set_model_status("Failed (regex-only)")
                    self._close_loading_dialog()
                    messagebox.showerror("Model Error", msg)
        except queue.Empty:
            pass
        try:
            while True:
                kind, payload = self.result_queue.get_nowait()
                self._finish_redaction()
                if kind == "redacted":
                    redacted, ml_error = payload
                    self.text_redacted.delete(1.0, tk.END)
                    self.text_redacted.insert(tk.END, redacted)
                    if ml_error:
                        messagebox.showwarning(
                            "Model Error",
                            "The NER model failed during redaction, so this output is regex-only "
                            f"(names and organisations may remain).\n\n{ml_error}",
                        )
                else:
                    messagebox.showerror("Redaction Error", f"Redaction failed:\n{payload}")
        except queue.Empty:
            pass
        # keep polling
        self.after(200, self._poll_queues)

    def _on_load_file(self):
        path = filedialog.askopenfilename(title="Open Text File", filetypes=[("Text Files", "*.txt")])
        if not path:
            return
        if not path.lower().endswith(".txt"):
            messagebox.showerror("Invalid File", "Only .txt files are supported.")
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = f.read()
            self.text_original.delete(1.0, tk.END)
            self.text_original.insert(tk.END, data)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{e}")

    def _on_redact(self):
        if self.redacting:
            return
        original = self.text_original.get(1.0, tk.END).rstrip()
        if not original:
            messagebox.showwarning("No Input", "Please enter or load text to redact.")
            return
        # Inference is slow on CPU (about half a second per line for BERT-large), so it runs
        # on a worker thread and posts the result back through result_queue.
        self.redacting = True
        self.redact_btn.state(["disabled"])
        self.load_btn.state(["disabled"])
        self.status_var.set("Redacting...")

        def worker():
            try:
                result = redact_text(original)
                self.result_queue.put(("redacted", (result, last_ml_error)))
            except Exception as exc:
                self.result_queue.put(("redact_error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_redaction(self):
        self.redacting = False
        self.redact_btn.state(["!disabled"])
        self.load_btn.state(["!disabled"])
        self.status_var.set(f"Model: {self.model_status}")

    def _on_save_file(self):
        redacted = self.text_redacted.get(1.0, tk.END).rstrip()
        if not redacted:
            messagebox.showwarning("Nothing to Save", "Redacted text is empty. Redact something first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt")],
            title="Save Redacted Text",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(redacted)
            messagebox.showinfo("Saved", f"Redacted text saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save file:\n{e}")


if __name__ == "__main__":
    app = PiiRedactorGUI()
    app.mainloop()
