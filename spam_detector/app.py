"""
SpamShield - modern desktop spam detector (customtkinter + scikit-learn).

Run:  python app.py
"""
from __future__ import annotations

import csv
import os
import queue
import re
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import customtkinter as ctk

import spam_engine as engine

# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# (light, dark) colour pairs
C_BG = ("#F3F4F8", "#0F1117")
C_SIDEBAR = ("#FFFFFF", "#151823")
C_CARD = ("#FFFFFF", "#1A1D29")
C_CARD_ALT = ("#EEF0F7", "#222638")
C_BORDER = ("#E2E5EF", "#2A2F45")
C_TEXT = ("#161A2B", "#ECEEF6")
C_MUTED = ("#6B7280", "#9AA1B9")
C_ACCENT = ("#5B5BF0", "#6D6DFF")
C_ACCENT_HOVER = ("#4848D8", "#5757E8")
C_SPAM = ("#DC2626", "#F87171")
C_SPAM_BG = ("#FDECEC", "#3A1D24")
C_HAM = ("#16A34A", "#4ADE80")
C_HAM_BG = ("#E7F7ED", "#173325")
C_WARN = ("#D97706", "#FBBF24")

FONT = "Segoe UI" if os.name == "nt" else ("SF Pro Display" if os.uname().sysname == "Darwin" else "DejaVu Sans")

SAMPLE_SPAM = ("WINNER!! As a valued network customer you have been selected to receive a £900 prize "
               "reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.")
SAMPLE_HAM = "Hey, are we still meeting for lunch tomorrow? Let me know what time works for you."


def f(size=13, weight="normal"):
    return ctk.CTkFont(family=FONT, size=size, weight=weight)


def pick(pair):
    """Resolve a (light, dark) pair for widgets that need a plain colour string."""
    return pair[1] if ctk.get_appearance_mode() == "Dark" else pair[0]


# --------------------------------------------------------------------------- #
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("SpamShield · AI Spam Detector")
        self.geometry("1220x780")
        self.minsize(1040, 700)
        self.configure(fg_color=C_BG)

        self.detector: engine.SpamDetector | None = None
        self.history: list[dict] = []
        self.batch_results: list[dict] = []
        self._ui_queue: queue.Queue = queue.Queue()
        self.threshold = tk.DoubleVar(value=0.50)
        self._busy_train = False

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self.content = ctk.CTkFrame(self, fg_color="transparent")
        self.content.grid(row=0, column=1, sticky="nsew", padx=(0, 18), pady=18)
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_rowconfigure(0, weight=1)

        self.pages: dict[str, ctk.CTkFrame] = {}
        self._build_scan_page()
        self._build_batch_page()
        self._build_insights_page()
        self._build_history_page()

        self.bind("<Control-Return>", lambda e: self.run_scan())
        self.after(100, self._poll_queue)
        self.show_page("scan")
        self._style_trees()
        self.after(200, self._load_or_train)

    # ------------------------------------------------------------------ #
    # infrastructure
    # ------------------------------------------------------------------ #
    def _poll_queue(self):
        try:
            while True:
                self._ui_queue.get_nowait()()
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def run_bg(self, work, done=None, error=None):
        """Run `work` in a thread; call done(result)/error(exc) on the UI thread."""
        def target():
            try:
                res = work()
            except Exception as exc:  # noqa: BLE001
                self._ui_queue.put(lambda exc=exc: (error or self._show_error)(exc))
                return
            if done:
                self._ui_queue.put(lambda res=res: done(res))
        threading.Thread(target=target, daemon=True).start()

    def _show_error(self, exc):
        messagebox.showerror("SpamShield", str(exc))

    def _load_or_train(self):
        if engine.model_exists():
            try:
                self.detector = engine.SpamDetector()
                self._on_model_ready()
                return
            except Exception:  # noqa: BLE001  (corrupt / incompatible model -> retrain)
                pass
        self.retrain()

    def _on_model_ready(self):
        m = self.detector.metrics
        if m:
            t = m["test"]
            self.model_badge.configure(
                text=f"{m['best_model']}\nAccuracy {t['accuracy']*100:.1f}%  ·  F1 {t['f1']*100:.1f}%")
        else:
            self.model_badge.configure(text="Model loaded")
        self.status_dot.configure(text_color=C_HAM)
        self.analyze_btn.configure(state="normal")
        self.batch_open_btn.configure(state="normal")
        self._refresh_insights()

    # ------------------------------------------------------------------ #
    # sidebar
    # ------------------------------------------------------------------ #
    def _build_sidebar(self):
        sb = ctk.CTkFrame(self, width=240, corner_radius=0, fg_color=C_SIDEBAR)
        sb.grid(row=0, column=0, sticky="nsew", padx=(0, 18))
        sb.grid_propagate(False)
        sb.grid_rowconfigure(6, weight=1)

        brand = ctk.CTkFrame(sb, fg_color="transparent")
        brand.grid(row=0, column=0, sticky="ew", padx=22, pady=(28, 4))
        ctk.CTkLabel(brand, text="🛡", font=f(26)).pack(side="left")
        ctk.CTkLabel(brand, text=" SpamShield", font=f(22, "bold"), text_color=C_TEXT).pack(side="left")
        ctk.CTkLabel(sb, text="AI-powered SMS & message\nspam detection", font=f(12),
                     text_color=C_MUTED, justify="left", anchor="w").grid(
            row=1, column=0, sticky="w", padx=24, pady=(0, 26))

        self.nav_buttons: dict[str, ctk.CTkButton] = {}
        items = [("scan", "🔍   Scan message"), ("batch", "📂   Batch scan"),
                 ("insights", "📊   Model insights"), ("history", "🕘   History")]
        for i, (key, label) in enumerate(items):
            b = ctk.CTkButton(sb, text=label, anchor="w", height=42, corner_radius=10,
                              font=f(14, "bold"), fg_color="transparent", text_color=C_MUTED,
                              hover_color=C_CARD_ALT, command=lambda k=key: self.show_page(k))
            b.grid(row=2 + i, column=0, sticky="ew", padx=14, pady=2)
            self.nav_buttons[key] = b

        bottom = ctk.CTkFrame(sb, fg_color="transparent")
        bottom.grid(row=7, column=0, sticky="sew", padx=18, pady=18)
        badge = ctk.CTkFrame(bottom, fg_color=C_CARD_ALT, corner_radius=12)
        badge.pack(fill="x", pady=(0, 14))
        head = ctk.CTkFrame(badge, fg_color="transparent")
        head.pack(fill="x", padx=14, pady=(10, 0))
        self.status_dot = ctk.CTkLabel(head, text="●", font=f(12), text_color=C_WARN)
        self.status_dot.pack(side="left")
        ctk.CTkLabel(head, text=" ACTIVE MODEL", font=f(10, "bold"), text_color=C_MUTED).pack(side="left")
        self.model_badge = ctk.CTkLabel(badge, text="Loading…", font=f(12), text_color=C_TEXT,
                                        justify="left", anchor="w")
        self.model_badge.pack(fill="x", padx=14, pady=(2, 12))

        self.theme_switch = ctk.CTkSwitch(bottom, text="Dark mode", font=f(13), text_color=C_MUTED,
                                          command=self._toggle_theme, progress_color=C_ACCENT)
        self.theme_switch.select()
        self.theme_switch.pack(anchor="w", padx=4)

    def show_page(self, key: str):
        for k, p in self.pages.items():
            if k == key:
                p.grid(row=0, column=0, sticky="nsew")
            else:
                p.grid_remove()
        for k, b in self.nav_buttons.items():
            active = k == key
            b.configure(fg_color=C_ACCENT if active else "transparent",
                        text_color=("#FFFFFF", "#FFFFFF") if active else C_MUTED,
                        hover_color=C_ACCENT_HOVER if active else C_CARD_ALT)
        if key == "history":
            self._refresh_history()

    def _toggle_theme(self):
        ctk.set_appearance_mode("dark" if self.theme_switch.get() else "light")
        self._style_trees()

    # ------------------------------------------------------------------ #
    # shared widgets
    # ------------------------------------------------------------------ #
    def _page(self, key: str, title: str, subtitle: str):
        page = ctk.CTkFrame(self.content, fg_color="transparent")
        page.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(page, text=title, font=f(28, "bold"), text_color=C_TEXT, anchor="w").grid(
            row=0, column=0, sticky="w", padx=4, pady=(6, 0))
        ctk.CTkLabel(page, text=subtitle, font=f(13), text_color=C_MUTED, anchor="w").grid(
            row=1, column=0, sticky="w", padx=4, pady=(0, 14))
        self.pages[key] = page
        return page

    def _card(self, parent, **kw):
        return ctk.CTkFrame(parent, fg_color=C_CARD, corner_radius=16, border_width=1,
                            border_color=C_BORDER, **kw)

    def _stat_card(self, parent, label, value="–", color=C_TEXT):
        card = ctk.CTkFrame(parent, fg_color=C_CARD, corner_radius=14, border_width=1, border_color=C_BORDER)
        ctk.CTkLabel(card, text=label.upper(), font=f(10, "bold"), text_color=C_MUTED).pack(
            anchor="w", padx=16, pady=(12, 0))
        val = ctk.CTkLabel(card, text=value, font=f(24, "bold"), text_color=color)
        val.pack(anchor="w", padx=16, pady=(0, 12))
        card.value_label = val
        return card

    def _style_trees(self):
        s = ttk.Style()
        s.theme_use("clam")
        bg, fg = pick(C_CARD), pick(C_TEXT)
        s.configure("Spam.Treeview", background=bg, fieldbackground=bg, foreground=fg,
                    rowheight=30, borderwidth=0, relief="flat", font=(FONT, 11),
                    bordercolor=bg, lightcolor=bg, darkcolor=bg)
        s.layout("Spam.Treeview", [("Treeview.treearea", {"sticky": "nswe"})])
        s.configure("Spam.Treeview.Heading", background=pick(C_CARD_ALT), foreground=pick(C_MUTED),
                    relief="flat", font=(FONT, 10, "bold"), padding=(8, 8))
        s.map("Spam.Treeview", background=[("selected", pick(C_ACCENT))],
              foreground=[("selected", "#FFFFFF")])
        s.map("Spam.Treeview.Heading", background=[("active", pick(C_CARD_ALT))])
        for tree in getattr(self, "_trees", []):
            tree.tag_configure("spam", foreground=pick(C_SPAM))
            tree.tag_configure("ham", foreground=pick(C_HAM))

    def _make_tree(self, parent, columns, widths, anchors=None):
        wrap = ctk.CTkFrame(parent, fg_color="transparent")
        wrap.grid_rowconfigure(0, weight=1)
        wrap.grid_columnconfigure(0, weight=1)
        tree = ttk.Treeview(wrap, columns=[c for c, _ in columns], show="headings",
                            style="Spam.Treeview", selectmode="browse")
        for (cid, text), w in zip(columns, widths):
            tree.heading(cid, text=text)
            tree.column(cid, width=w, minwidth=50, stretch=(cid == columns[-1][0]),
                        anchor=(anchors or {}).get(cid, "w"))
        sb = ctk.CTkScrollbar(wrap, command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns", padx=(4, 0))
        if not hasattr(self, "_trees"):
            self._trees = []
        self._trees.append(tree)
        tree.tag_configure("spam", foreground=pick(C_SPAM))
        tree.tag_configure("ham", foreground=pick(C_HAM))
        return wrap, tree

    # ------------------------------------------------------------------ #
    # PAGE 1 - scan a single message
    # ------------------------------------------------------------------ #
    def _build_scan_page(self):
        page = self._page("scan", "Scan a message",
                          "Paste any SMS, email or chat message — the model tells you if it's spam and explains why.")
        page.grid_rowconfigure(2, weight=1)
        body = ctk.CTkFrame(page, fg_color="transparent")
        body.grid(row=2, column=0, sticky="nsew")
        body.grid_columnconfigure(0, weight=5, uniform="c")
        body.grid_columnconfigure(1, weight=4, uniform="c")
        body.grid_rowconfigure(0, weight=1)

        # ---- input card
        left = self._card(body)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(left, text="Message", font=f(14, "bold"), text_color=C_TEXT).grid(
            row=0, column=0, sticky="w", padx=20, pady=(18, 6))
        self.input = ctk.CTkTextbox(left, font=f(15), corner_radius=12, wrap="word",
                                    fg_color=C_CARD_ALT, text_color=C_TEXT, border_width=0)
        self.input.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 8))
        self.input.bind("<KeyRelease>", self._on_input_change)
        self._tk_input = self.input._textbox

        meta = ctk.CTkFrame(left, fg_color="transparent")
        meta.grid(row=2, column=0, sticky="ew", padx=20)
        self.counter = ctk.CTkLabel(meta, text="0 characters · 0 words", font=f(12), text_color=C_MUTED)
        self.counter.pack(side="left")
        ctk.CTkButton(meta, text="Normal example", width=120, height=28, font=f(12), corner_radius=8,
                      fg_color=C_CARD_ALT, text_color=C_TEXT, hover_color=C_BORDER,
                      command=lambda: self._set_input(SAMPLE_HAM)).pack(side="right")
        ctk.CTkButton(meta, text="Spam example", width=110, height=28, font=f(12), corner_radius=8,
                      fg_color=C_CARD_ALT, text_color=C_TEXT, hover_color=C_BORDER,
                      command=lambda: self._set_input(SAMPLE_SPAM)).pack(side="right", padx=6)

        sens = ctk.CTkFrame(left, fg_color="transparent")
        sens.grid(row=3, column=0, sticky="ew", padx=20, pady=(14, 4))
        sens.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(sens, text="Sensitivity", font=f(13, "bold"), text_color=C_TEXT).grid(row=0, column=0, sticky="w")
        self.sens_value = ctk.CTkLabel(sens, text="Balanced (50%)", font=f(12), text_color=C_MUTED)
        self.sens_value.grid(row=0, column=2, sticky="e")
        ctk.CTkSlider(sens, from_=0.10, to=0.90, number_of_steps=16, variable=self.threshold,
                      command=self._on_sens, progress_color=C_ACCENT, button_color=C_ACCENT,
                      button_hover_color=C_ACCENT_HOVER).grid(row=0, column=1, sticky="ew", padx=14)
        ctk.CTkLabel(sens, text="Lower = flags more messages as spam · Higher = only flags when very sure",
                     font=f(11), text_color=C_MUTED).grid(row=1, column=0, columnspan=3, sticky="w", pady=(2, 0))

        btns = ctk.CTkFrame(left, fg_color="transparent")
        btns.grid(row=4, column=0, sticky="ew", padx=20, pady=(8, 20))
        btns.grid_columnconfigure(0, weight=1)
        self.analyze_btn = ctk.CTkButton(btns, text="Analyze message   (Ctrl+Enter)", height=46, corner_radius=12,
                                         font=f(15, "bold"), fg_color=C_ACCENT, hover_color=C_ACCENT_HOVER,
                                         state="disabled", command=self.run_scan)
        self.analyze_btn.grid(row=0, column=0, sticky="ew")
        ctk.CTkButton(btns, text="Clear", width=90, height=46, corner_radius=12, font=f(14),
                      fg_color=C_CARD_ALT, text_color=C_TEXT, hover_color=C_BORDER,
                      command=lambda: self._set_input("")).grid(row=0, column=1, padx=(10, 0))

        # ---- result card
        right = self._card(body)
        right.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        right.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(right, text="Result", font=f(14, "bold"), text_color=C_TEXT).grid(
            row=0, column=0, sticky="w", padx=20, pady=(18, 8))

        self.verdict_box = ctk.CTkFrame(right, fg_color=C_CARD_ALT, corner_radius=14)
        self.verdict_box.grid(row=1, column=0, sticky="ew", padx=20)
        self.verdict_icon = ctk.CTkLabel(self.verdict_box, text="🛡", font=f(34))
        self.verdict_icon.grid(row=0, column=0, rowspan=2, padx=(18, 10), pady=14)
        self.verdict_label = ctk.CTkLabel(self.verdict_box, text="Waiting for a message", font=f(21, "bold"),
                                          text_color=C_TEXT, anchor="w")
        self.verdict_label.grid(row=0, column=1, sticky="w", pady=(14, 0))
        self.risk_label = ctk.CTkLabel(self.verdict_box, text="Paste something and press Analyze",
                                       font=f(12), text_color=C_MUTED, anchor="w")
        self.risk_label.grid(row=1, column=1, sticky="w", pady=(0, 14))

        prob = ctk.CTkFrame(right, fg_color="transparent")
        prob.grid(row=2, column=0, sticky="ew", padx=20, pady=(16, 0))
        prob.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(prob, text="Spam probability", font=f(12), text_color=C_MUTED).grid(row=0, column=0, sticky="w")
        self.prob_value = ctk.CTkLabel(prob, text="–", font=f(18, "bold"), text_color=C_TEXT)
        self.prob_value.grid(row=0, column=1, sticky="e")
        self.prob_bar = ctk.CTkProgressBar(prob, height=12, corner_radius=6, progress_color=C_ACCENT,
                                           fg_color=C_CARD_ALT)
        self.prob_bar.set(0)
        self.prob_bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        self.why = ctk.CTkScrollableFrame(right, fg_color="transparent", label_text="")
        self.why.grid(row=3, column=0, sticky="nsew", padx=(14, 8), pady=(10, 14))
        self.why.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(3, weight=1)

        self._render_why(None)

    def _on_sens(self, _=None):
        t = self.threshold.get()
        name = "Very sensitive" if t <= 0.25 else "Sensitive" if t < 0.45 else \
            "Balanced" if t <= 0.6 else "Strict" if t < 0.8 else "Very strict"
        self.sens_value.configure(text=f"{name} ({t*100:.0f}%)")

    def _on_input_change(self, _=None):
        txt = self.input.get("1.0", "end-1c")
        self.counter.configure(text=f"{len(txt)} characters · {len(txt.split())} words")
        self._tk_input.tag_remove("spamword", "1.0", "end")

    def _set_input(self, text: str):
        self.input.delete("1.0", "end")
        self.input.insert("1.0", text)
        self._on_input_change()
        self._reset_result()
        self.input.focus_set()

    def _reset_result(self):
        self.verdict_box.configure(fg_color=C_CARD_ALT)
        self.verdict_icon.configure(text="🛡")
        self.verdict_label.configure(text="Waiting for a message", text_color=C_TEXT)
        self.risk_label.configure(text="Paste something and press Analyze")
        self.prob_value.configure(text="–", text_color=C_TEXT)
        self.prob_bar.set(0)
        self.prob_bar.configure(progress_color=C_ACCENT)
        self._render_why(None)

    def _section(self, parent, title):
        ctk.CTkLabel(parent, text=title.upper(), font=f(10, "bold"), text_color=C_MUTED, anchor="w").pack(
            fill="x", padx=6, pady=(10, 4))

    def _render_why(self, r: dict | None):
        for w in self.why.winfo_children():
            w.destroy()
        if r is None:
            ctk.CTkLabel(self.why, text="The explanation appears here:\nwhich words and patterns pushed the score "
                                        "towards spam or towards a normal message.",
                         font=f(12), text_color=C_MUTED, justify="left", anchor="w", wraplength=300).pack(
                fill="x", padx=6, pady=10)
            return

        if r["spam_words"]:
            self._section(self.why, "Pushes towards spam")
            top = max(v for _, v in r["spam_words"])
            for word, val in r["spam_words"]:
                self._bar_row(self.why, word, val / top, C_SPAM, f"+{val:.1f}")
        if r["ham_words"]:
            self._section(self.why, "Looks like a normal message")
            top = max(-v for _, v in r["ham_words"])
            for word, val in r["ham_words"]:
                self._bar_row(self.why, word, -val / top, C_HAM, f"{val:.1f}")
        self._section(self.why, "Red flags")
        if r["flags"]:
            for fl in r["flags"]:
                row = ctk.CTkFrame(self.why, fg_color=C_SPAM_BG, corner_radius=8)
                row.pack(fill="x", padx=4, pady=2)
                ctk.CTkLabel(row, text="⚠  " + fl, font=f(12), text_color=C_SPAM, anchor="w",
                             justify="left", wraplength=300).pack(fill="x", padx=10, pady=6)
        else:
            row = ctk.CTkFrame(self.why, fg_color=C_HAM_BG, corner_radius=8)
            row.pack(fill="x", padx=4, pady=2)
            ctk.CTkLabel(row, text="✓  No common spam patterns found", font=f(12), text_color=C_HAM,
                         anchor="w").pack(fill="x", padx=10, pady=6)
        ctk.CTkLabel(self.why, text="Word impact = change in the model's log-odds when that word is removed.",
                     font=f(10), text_color=C_MUTED, anchor="w", justify="left", wraplength=300).pack(
            fill="x", padx=6, pady=(12, 4))

    def _bar_row(self, parent, word, frac, color, number):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=4, pady=1)
        row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(row, text=word[:18], font=f(12), text_color=C_TEXT, width=110, anchor="w").grid(row=0, column=0)
        bar = ctk.CTkProgressBar(row, height=8, corner_radius=4, progress_color=color, fg_color=C_CARD_ALT)
        bar.set(max(0.05, min(1.0, frac)))
        bar.grid(row=0, column=1, sticky="ew", padx=8)
        ctk.CTkLabel(row, text=number, font=f(11), text_color=C_MUTED, width=36, anchor="e").grid(row=0, column=2)

    def run_scan(self):
        if self.detector is None:
            return
        text = self.input.get("1.0", "end-1c").strip()
        if not text:
            self.risk_label.configure(text="Please type or paste a message first.")
            return
        r = self.detector.analyze(text, threshold=self.threshold.get())
        p = r["prob"]
        spam = r["is_spam"]
        col, bg = (C_SPAM, C_SPAM_BG) if spam else (C_HAM, C_HAM_BG)
        self.verdict_box.configure(fg_color=bg)
        self.verdict_icon.configure(text="🚨" if spam else "✅")
        self.verdict_label.configure(text="SPAM detected" if spam else "Looks safe (ham)", text_color=col)
        self.risk_label.configure(text=f"{r['risk']}  ·  threshold {self.threshold.get()*100:.0f}%")
        self.prob_value.configure(text=f"{p*100:.1f}%", text_color=col)
        self.prob_bar.configure(progress_color=col)
        self.prob_bar.set(p)
        self._render_why(r)
        self._highlight(r)
        self.history.insert(0, {"time": time.strftime("%H:%M:%S"), "prob": p, "is_spam": spam, "text": text})
        self.history = self.history[:500]

    def _highlight(self, r):
        t = self._tk_input
        t.tag_remove("spamword", "1.0", "end")
        t.tag_configure("spamword", background=pick(C_SPAM_BG), foreground=pick(C_SPAM), underline=True)
        content = t.get("1.0", "end-1c")
        for word, _ in r["spam_words"]:
            for m in re.finditer(re.escape(word), content, re.I):
                t.tag_add("spamword", f"1.0+{m.start()}c", f"1.0+{m.end()}c")

    # ------------------------------------------------------------------ #
    # PAGE 2 - batch scan
    # ------------------------------------------------------------------ #
    def _build_batch_page(self):
        page = self._page("batch", "Batch scan",
                          "Scan a whole file at once — a CSV with a message column, or a .txt with one message per line.")
        page.grid_rowconfigure(4, weight=1)

        bar = ctk.CTkFrame(page, fg_color="transparent")
        bar.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        self.batch_open_btn = ctk.CTkButton(bar, text="📂  Open file…", height=40, corner_radius=10,
                                            font=f(14, "bold"), fg_color=C_ACCENT, hover_color=C_ACCENT_HOVER,
                                            state="disabled", command=self.batch_open)
        self.batch_open_btn.pack(side="left")
        self.batch_export_btn = ctk.CTkButton(bar, text="Export results…", height=40, corner_radius=10,
                                              font=f(14), fg_color=C_CARD, text_color=C_TEXT, border_width=1,
                                              border_color=C_BORDER, hover_color=C_CARD_ALT, state="disabled",
                                              command=self.batch_export)
        self.batch_export_btn.pack(side="left", padx=10)
        self.batch_file_label = ctk.CTkLabel(bar, text="No file loaded", font=f(12), text_color=C_MUTED)
        self.batch_file_label.pack(side="left", padx=8)
        self.batch_filter = ctk.CTkSegmentedButton(bar, values=["All", "Spam", "Ham"], command=self._batch_render,
                                                   selected_color=C_ACCENT, selected_hover_color=C_ACCENT_HOVER)
        self.batch_filter.set("All")
        self.batch_filter.pack(side="right")

        stats = ctk.CTkFrame(page, fg_color="transparent")
        stats.grid(row=3, column=0, sticky="ew", pady=(0, 12))
        for i in range(4):
            stats.grid_columnconfigure(i, weight=1, uniform="s")
        self.b_total = self._stat_card(stats, "Messages scanned")
        self.b_spam = self._stat_card(stats, "Spam found", color=C_SPAM)
        self.b_ham = self._stat_card(stats, "Safe (ham)", color=C_HAM)
        self.b_rate = self._stat_card(stats, "Spam rate")
        for i, c in enumerate((self.b_total, self.b_spam, self.b_ham, self.b_rate)):
            c.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 6, 0 if i == 3 else 6))

        card = self._card(page)
        card.grid(row=4, column=0, sticky="nsew")
        card.grid_rowconfigure(1, weight=1)
        card.grid_columnconfigure(0, weight=1)
        self.batch_progress = ctk.CTkProgressBar(card, height=6, progress_color=C_ACCENT, fg_color=C_CARD_ALT)
        self.batch_progress.set(0)
        self.batch_progress.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 8))
        wrap, self.batch_tree = self._make_tree(
            card, [("n", "#"), ("verdict", "VERDICT"), ("prob", "SPAM %"), ("text", "MESSAGE")],
            [60, 90, 90, 600], anchors={"n": "center", "verdict": "center", "prob": "center"})
        wrap.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))

    @staticmethod
    def _read_messages(path: str) -> tuple[list[str], str]:
        import pandas as pd
        ext = os.path.splitext(path)[1].lower()
        if ext in (".txt", ".log"):
            for enc in ("utf-8", "cp1252"):
                try:
                    with open(path, "r", encoding=enc) as fh:
                        lines = [ln.strip() for ln in fh if ln.strip()]
                    return lines, "one message per line"
                except UnicodeDecodeError:
                    continue
            raise ValueError("Could not decode the text file.")
        df = None
        for enc in ("utf-8", "cp1252", "latin-1"):
            try:
                df = pd.read_csv(path, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        if df is None:
            raise ValueError("Could not read the CSV file.")
        lower = {str(c).lower().strip(): c for c in df.columns}
        col = next((lower[k] for k in ("v2", "text", "message", "sms", "email", "body", "content") if k in lower), None)
        if col is None:
            obj = [c for c in df.columns if df[c].dtype == object]
            if not obj:
                raise ValueError("No text column found in this CSV.")
            col = max(obj, key=lambda c: df[c].astype(str).str.len().mean())
        return [engine.fix_encoding(t) for t in df[col].dropna().astype(str)], f"column “{col}”"

    def batch_open(self):
        if self.detector is None:
            return
        path = filedialog.askopenfilename(title="Choose a file to scan",
                                          filetypes=[("CSV / text", "*.csv *.txt"), ("All files", "*.*")])
        if not path:
            return
        self.batch_open_btn.configure(state="disabled")
        self.batch_file_label.configure(text="Scanning…")
        self.batch_progress.configure(mode="indeterminate")
        self.batch_progress.start()
        thr = self.threshold.get()
        det = self.detector

        def work():
            msgs, how = self._read_messages(path)
            return det.analyze_many(msgs, threshold=thr), how

        def done(res):
            results, how = res
            self.batch_progress.stop()
            self.batch_progress.configure(mode="determinate")
            self.batch_progress.set(1)
            self.batch_results = results
            self.batch_open_btn.configure(state="normal")
            self.batch_export_btn.configure(state="normal" if results else "disabled")
            self.batch_file_label.configure(text=f"{os.path.basename(path)} · {how}")
            n_spam = sum(r["is_spam"] for r in results)
            n = len(results)
            self.b_total.value_label.configure(text=f"{n:,}")
            self.b_spam.value_label.configure(text=f"{n_spam:,}")
            self.b_ham.value_label.configure(text=f"{n - n_spam:,}")
            self.b_rate.value_label.configure(text=f"{(n_spam / n * 100) if n else 0:.1f}%")
            self._batch_render()

        def err(exc):
            self.batch_progress.stop()
            self.batch_progress.configure(mode="determinate")
            self.batch_progress.set(0)
            self.batch_open_btn.configure(state="normal")
            self.batch_file_label.configure(text="Failed to load file")
            self._show_error(exc)

        self.run_bg(work, done, err)

    def _batch_render(self, _=None):
        tree = self.batch_tree
        tree.delete(*tree.get_children())
        flt = self.batch_filter.get()
        shown = 0
        for i, r in enumerate(self.batch_results, 1):
            if flt == "Spam" and not r["is_spam"]:
                continue
            if flt == "Ham" and r["is_spam"]:
                continue
            tree.insert("", "end", tags=("spam" if r["is_spam"] else "ham",),
                        values=(i, "SPAM" if r["is_spam"] else "HAM", f"{r['prob']*100:.1f}%",
                                r["text"].replace("\n", " ")[:300]))
            shown += 1
            if shown >= 5000:
                break

    def batch_export(self):
        if not self.batch_results:
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="spam_scan_results.csv",
                                            filetypes=[("CSV", "*.csv")])
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["verdict", "spam_probability", "risk", "message"])
            for r in self.batch_results:
                w.writerow(["spam" if r["is_spam"] else "ham", f"{r['prob']:.4f}", r["risk"], r["text"]])
        messagebox.showinfo("SpamShield", f"Saved {len(self.batch_results):,} rows to\n{path}")

    # ------------------------------------------------------------------ #
    # PAGE 3 - model insights
    # ------------------------------------------------------------------ #
    def _build_insights_page(self):
        page = self._page("insights", "Model insights",
                          "How the AI was trained and how well it performs on messages it has never seen.")
        page.grid_rowconfigure(2, weight=1)
        sc = ctk.CTkScrollableFrame(page, fg_color="transparent")
        sc.grid(row=2, column=0, sticky="nsew")
        sc.grid_columnconfigure(0, weight=1)
        self.ins = sc

        top = ctk.CTkFrame(sc, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew")
        for i in range(5):
            top.grid_columnconfigure(i, weight=1, uniform="m")
        self.m_cards = {}
        for i, (k, lbl, col) in enumerate([("accuracy", "Accuracy", C_TEXT), ("precision", "Precision", C_TEXT),
                                           ("recall", "Recall", C_TEXT), ("f1", "F1 score", C_ACCENT),
                                           ("roc_auc", "ROC-AUC", C_TEXT)]):
            c = self._stat_card(top, lbl, color=col)
            c.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 5, 0 if i == 4 else 5))
            self.m_cards[k] = c

        mid = ctk.CTkFrame(sc, fg_color="transparent")
        mid.grid(row=1, column=0, sticky="ew", pady=12)
        mid.grid_columnconfigure(0, weight=2, uniform="x")
        mid.grid_columnconfigure(1, weight=3, uniform="x")

        self.cm_card = self._card(mid)
        self.cm_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.cmp_card = self._card(mid)
        self.cmp_card.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        words = ctk.CTkFrame(sc, fg_color="transparent")
        words.grid(row=2, column=0, sticky="ew")
        words.grid_columnconfigure((0, 1), weight=1, uniform="w")
        self.spamw_card = self._card(words)
        self.spamw_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.hamw_card = self._card(words)
        self.hamw_card.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        self.train_card = self._card(sc)
        self.train_card.grid(row=3, column=0, sticky="ew", pady=12)

    def _refresh_insights(self):
        m = self.detector.metrics if self.detector else {}
        for w in (self.cm_card, self.cmp_card, self.spamw_card, self.hamw_card, self.train_card):
            for c in w.winfo_children():
                c.destroy()
        if not m:
            ctk.CTkLabel(self.train_card, text="No training report yet — click “Retrain model”.",
                         font=f(13), text_color=C_MUTED).pack(padx=20, pady=20)
            self._build_train_controls()
            return
        t = m["test"]
        for k, c in self.m_cards.items():
            c.value_label.configure(text=f"{t[k]*100:.2f}%")

        # confusion matrix
        cm = m["confusion"]
        ctk.CTkLabel(self.cm_card, text="Confusion matrix (held-out test set)", font=f(14, "bold"),
                     text_color=C_TEXT).pack(anchor="w", padx=20, pady=(16, 2))
        ctk.CTkLabel(self.cm_card, text=f"{m['test_size']:,} unseen messages", font=f(11),
                     text_color=C_MUTED).pack(anchor="w", padx=20)
        grid = ctk.CTkFrame(self.cm_card, fg_color="transparent")
        grid.pack(padx=20, pady=14, fill="x")
        grid.grid_columnconfigure((1, 2), weight=1, uniform="g")
        ctk.CTkLabel(grid, text="", width=70).grid(row=0, column=0)
        ctk.CTkLabel(grid, text="Predicted HAM", font=f(11), text_color=C_MUTED).grid(row=0, column=1)
        ctk.CTkLabel(grid, text="Predicted SPAM", font=f(11), text_color=C_MUTED).grid(row=0, column=2)
        for r, (lbl, a, b, ca, cb) in enumerate([
                ("Actual HAM", cm["tn"], cm["fp"], C_HAM_BG, C_SPAM_BG),
                ("Actual SPAM", cm["fn"], cm["tp"], C_SPAM_BG, C_HAM_BG)], start=1):
            ctk.CTkLabel(grid, text=lbl, font=f(11), text_color=C_MUTED, width=80, anchor="w").grid(row=r, column=0)
            for ci, (val, bgc, good) in enumerate([(a, ca, r == 1), (b, cb, r == 2)], start=1):
                cell = ctk.CTkFrame(grid, fg_color=bgc, corner_radius=12, height=78)
                cell.grid(row=r, column=ci, padx=4, pady=4, sticky="nsew")
                cell.grid_propagate(False)
                cell.grid_columnconfigure(0, weight=1)
                cell.grid_rowconfigure((0, 1), weight=1)
                is_correct = (r == 1 and ci == 1) or (r == 2 and ci == 2)
                ctk.CTkLabel(cell, text=f"{val:,}", font=f(22, "bold"),
                             text_color=C_HAM if is_correct else C_SPAM).grid(row=0, column=0, sticky="s")
                ctk.CTkLabel(cell, text="correct" if is_correct else
                             ("false alarm" if r == 1 else "missed spam"), font=f(10),
                             text_color=C_MUTED).grid(row=1, column=0, sticky="n")
        ctk.CTkLabel(self.cm_card, text=f"Only {cm['fp']} normal message(s) wrongly flagged and {cm['fn']} spam missed.",
                     font=f(11), text_color=C_MUTED, wraplength=300, justify="left").pack(anchor="w", padx=20, pady=(0, 16))

        # model comparison
        ctk.CTkLabel(self.cmp_card, text="Model comparison", font=f(14, "bold"), text_color=C_TEXT).pack(
            anchor="w", padx=20, pady=(16, 2))
        ctk.CTkLabel(self.cmp_card, text="Model chosen by 5-fold cross-validation on the training data only",
                     font=f(11), text_color=C_MUTED).pack(anchor="w", padx=20)
        tbl = ctk.CTkFrame(self.cmp_card, fg_color="transparent")
        tbl.pack(fill="x", padx=20, pady=12)
        heads = ["Model", "CV F1", "Accuracy", "Precision", "Recall", "F1"]
        for ci, h in enumerate(heads):
            tbl.grid_columnconfigure(ci, weight=3 if ci == 0 else 1)
            ctk.CTkLabel(tbl, text=h.upper(), font=f(10, "bold"), text_color=C_MUTED,
                         anchor="w" if ci == 0 else "e").grid(row=0, column=ci, sticky="ew", pady=(0, 6))
        for ri, row in enumerate(m["comparison"], start=1):
            best = row["model"] == m["best_model"]
            vals = [row["model"] + ("  ★" if best else ""), f"{row['cv_f1_mean']*100:.2f}",
                    f"{row['accuracy']*100:.2f}", f"{row['precision']*100:.2f}",
                    f"{row['recall']*100:.2f}", f"{row['f1']*100:.2f}"]
            for ci, v in enumerate(vals):
                ctk.CTkLabel(tbl, text=v, font=f(13, "bold" if best else "normal"),
                             text_color=C_ACCENT if best else C_TEXT,
                             anchor="w" if ci == 0 else "e").grid(row=ri, column=ci, sticky="ew", pady=5)
        d = m["dataset"]
        ctk.CTkLabel(self.cmp_card,
                     text=(f"Dataset: {d['rows_clean']:,} unique messages ({d['n_spam']:,} spam / {d['n_ham']:,} ham) · "
                           f"{d['duplicates_removed']} duplicates removed\n"
                           f"Features: TF-IDF word 1-2-grams + character 3-5-grams · 80/20 stratified split"),
                     font=f(11), text_color=C_MUTED, justify="left", anchor="w", wraplength=520).pack(
            anchor="w", padx=20, pady=(0, 16))

        # top words
        for card, title, data, col in ((self.spamw_card, "Top spam indicators", m["top_spam_words"], C_SPAM),
                                       (self.hamw_card, "Top normal-message indicators", m["top_ham_words"], C_HAM)):
            ctk.CTkLabel(card, text=title, font=f(14, "bold"), text_color=C_TEXT).pack(anchor="w", padx=20, pady=(16, 8))
            mx = max(abs(v) for _, v in data) or 1
            for word, val in data[:10]:
                row = ctk.CTkFrame(card, fg_color="transparent")
                row.pack(fill="x", padx=20, pady=2)
                row.grid_columnconfigure(1, weight=1)
                ctk.CTkLabel(row, text=word, font=f(12), text_color=C_TEXT, width=100, anchor="w").grid(row=0, column=0)
                pb = ctk.CTkProgressBar(row, height=8, corner_radius=4, progress_color=col, fg_color=C_CARD_ALT)
                pb.set(abs(val) / mx)
                pb.grid(row=0, column=1, sticky="ew", padx=(6, 0))
            ctk.CTkLabel(card, text="", height=8).pack()

        self._build_train_controls()

    def _build_train_controls(self):
        card = self.train_card
        for c in card.winfo_children():
            c.destroy()
        m = self.detector.metrics if self.detector else {}
        ctk.CTkLabel(card, text="Training", font=f(14, "bold"), text_color=C_TEXT).grid(
            row=0, column=0, sticky="w", padx=20, pady=(16, 0))
        info = (f"Trained on {m.get('trained_on', '–')} · {m.get('trained_at', '–')} · {m.get('train_seconds', '–')}s"
                if m else "Model has not been trained yet.")
        ctk.CTkLabel(card, text=info, font=f(12), text_color=C_MUTED).grid(row=1, column=0, sticky="w", padx=20)
        card.grid_columnconfigure(0, weight=1)
        self.train_status = ctk.CTkLabel(card, text="", font=f(12), text_color=C_MUTED)
        self.train_status.grid(row=2, column=0, sticky="w", padx=20, pady=(6, 14))
        btns = ctk.CTkFrame(card, fg_color="transparent")
        btns.grid(row=0, column=1, rowspan=3, padx=20)
        self.retrain_btn = ctk.CTkButton(btns, text="⟳  Retrain model", height=38, corner_radius=10, font=f(13, "bold"),
                                         fg_color=C_ACCENT, hover_color=C_ACCENT_HOVER, command=self.retrain)
        self.retrain_btn.pack(side="left")
        self.train_other_btn = ctk.CTkButton(btns, text="Train on another CSV…", height=38, corner_radius=10, font=f(13),
                                             fg_color=C_CARD_ALT, text_color=C_TEXT, hover_color=C_BORDER,
                                             command=self.retrain_custom)
        self.train_other_btn.pack(side="left", padx=(8, 0))

    def retrain_custom(self):
        path = filedialog.askopenfilename(title="Labelled CSV (needs label + text columns)",
                                          filetypes=[("CSV", "*.csv")])
        if path:
            self.retrain(path)

    def retrain(self, csv_path: str | None = None):
        if self._busy_train:
            return
        self._busy_train = True
        csv_path = csv_path or engine.DATA_PATH
        self.status_dot.configure(text_color=C_WARN)
        self.model_badge.configure(text="Training model…\nthis takes ~15 seconds")
        self.analyze_btn.configure(state="disabled")
        if hasattr(self, "retrain_btn") and self.retrain_btn.winfo_exists():
            self.retrain_btn.configure(state="disabled", text="Training…")
            self.train_other_btn.configure(state="disabled")
            self.train_status.configure(text="Training and evaluating 3 models with cross-validation…")

        def work():
            import train_model
            train_model.train(csv_path, log=lambda *_: None)
            return engine.SpamDetector()

        def done(det):
            self._busy_train = False
            self.detector = det
            self._on_model_ready()

        def err(exc):
            self._busy_train = False
            self.status_dot.configure(text_color=C_SPAM)
            self.model_badge.configure(text="Training failed")
            if self.detector is not None:
                self._on_model_ready()
            self._refresh_insights()
            self._show_error(f"Training failed:\n{exc}")

        self.run_bg(work, done, err)

    # ------------------------------------------------------------------ #
    # PAGE 4 - history
    # ------------------------------------------------------------------ #
    def _build_history_page(self):
        page = self._page("history", "History", "Messages you checked in this session.")
        page.grid_rowconfigure(3, weight=1)
        bar = ctk.CTkFrame(page, fg_color="transparent")
        bar.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        ctk.CTkButton(bar, text="Clear history", height=36, corner_radius=10, font=f(13), fg_color=C_CARD,
                      text_color=C_TEXT, border_width=1, border_color=C_BORDER, hover_color=C_CARD_ALT,
                      command=self._clear_history).pack(side="left")
        self.hist_count = ctk.CTkLabel(bar, text="", font=f(12), text_color=C_MUTED)
        self.hist_count.pack(side="left", padx=12)
        card = self._card(page)
        card.grid(row=3, column=0, sticky="nsew")
        card.grid_rowconfigure(0, weight=1)
        card.grid_columnconfigure(0, weight=1)
        wrap, self.hist_tree = self._make_tree(
            card, [("time", "TIME"), ("verdict", "VERDICT"), ("prob", "SPAM %"), ("text", "MESSAGE")],
            [90, 90, 90, 600], anchors={"time": "center", "verdict": "center", "prob": "center"})
        wrap.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        self.hist_tree.bind("<Double-1>", self._history_reopen)

    def _refresh_history(self):
        tree = self.hist_tree
        tree.delete(*tree.get_children())
        for h in self.history:
            tree.insert("", "end", tags=("spam" if h["is_spam"] else "ham",),
                        values=(h["time"], "SPAM" if h["is_spam"] else "HAM", f"{h['prob']*100:.1f}%",
                                h["text"].replace("\n", " ")[:300]))
        self.hist_count.configure(text=f"{len(self.history)} checked · double-click a row to re-open it")

    def _history_reopen(self, _):
        sel = self.hist_tree.selection()
        if not sel:
            return
        idx = self.hist_tree.index(sel[0])
        if idx < len(self.history):
            self.show_page("scan")
            self._set_input(self.history[idx]["text"])
            self.run_scan()

    def _clear_history(self):
        self.history.clear()
        self._refresh_history()


if __name__ == "__main__":
    App().mainloop()
