"""Python-owned experiment state; the frontend sends editing gestures only."""

from pathlib import Path

import anywidget
import numpy as np
import traitlets as tl

from .lfsr import DEFAULT_TERMS, encode, integer_parameter, max_availability, polynomial_terms


class LLREditor(anywidget.AnyWidget):
    _esm = Path(__file__).with_name("editor.js")
    _css = Path(__file__).with_name("editor.css")
    state = tl.Dict().tag(sync=True)

    def __init__(self, n=16, iterations=20, terms=DEFAULT_TERMS, seed=None):
        terms = polynomial_terms(terms)
        n = integer_parameter(n, "N", 1)
        iterations = integer_parameter(iterations, "T")
        # Anywidget's class-level file objects can retain the imported contents.
        # Read assets for each new editor so rerunning the cell picks up edits.
        assets = Path(__file__).parent
        super().__init__(
            _esm=(assets / "editor.js").read_text(encoding="utf-8"),
            _css=(assets / "editor.css").read_text(encoding="utf-8"),
        )
        self.layout.width = "100%"
        self.layout.margin = "0"
        self.layout.padding = "0"
        self.layout.border = "0"
        self.terms = terms
        self.rng = np.random.default_rng(seed)
        self.n = n
        self.iterations = iterations
        self.polynomial = " + ".join(
            "1" if power == 0 else "x" if power == 1 else f"x^{power}" for power in reversed(terms)
        )
        self.x = self.rng.integers(0, 2, terms[-1], dtype=np.uint8)
        self._colors = []
        self.availability = max_availability(n, terms)
        self._encode()
        self.on_msg(self._message)

    def _encode(self):
        self.y = encode(self.x, self.n, self.terms)
        self.truth_llr = 1.0 - 2.0 * self.y
        self.y_prime = self.truth_llr.copy()
        self._code_state = dict(
            polynomial=self.polynomial,
            n=self.n,
            availability=self.availability,
            x=self.x.tolist(),
            y=self.y.tolist(),
            channel_llr=self.truth_llr.tolist(),
        )
        self._publish()

    def _publish(self, error="", index=None):
        selection = slice(None) if index is None else slice(index, index + 1)
        aligned = self.truth_llr[selection] * self.y_prime[selection]
        # Fixed scale across edits; zero is exactly white.
        strength = -np.expm1(-np.abs(aligned)[:, None] / 2)
        targets = np.where((aligned >= 0)[:, None], [45, 170, 92], [225, 68, 68])
        rgb = np.rint(255 + strength * (targets - 255)).astype(np.uint8)
        colors = [f"rgb({r},{g},{b})" for r, g, b in rgb]
        if index is None:
            self._colors = colors
        else:
            self._colors[index] = colors[0]
        self.state = dict(
            **self._code_state,
            iterations=self.iterations,
            edited=self.y_prime.tolist(),
            colors=self._colors.copy(),
            error=error,
        )

    def snapshot(self):
        """Inputs for decode(**editor.snapshot()); taking a snapshot runs no decoder."""
        return dict(
            initial=self.y_prime.copy(),
            truth=self.y.copy(),
            terms=self.terms,
            iterations=self.iterations,
        )

    def _message(self, widget, event, buffers):
        try:
            self.apply(event)
        except (ValueError, IndexError, KeyError, TypeError, OverflowError) as exc:
            self._publish(str(exc))

    def apply(self, event):
        """Apply a frontend command, also useful for reproducible scripted experiments."""
        action = event["action"]
        index = None
        if action == "configure":
            n = integer_parameter(int(event["n"]), "N", 1)
            t = integer_parameter(int(event["iterations"]), "T")
            changed = n != self.n
            self.n, self.iterations = n, t
            if changed:
                self.availability = max_availability(n, self.terms)
                self._encode()
                return
        elif action in ("random", "flip_x"):
            if action == "random":
                self.x = self.rng.integers(0, 2, len(self.x), dtype=np.uint8)
            else:
                i = int(event["index"])
                if not 0 <= i < len(self.x):
                    raise ValueError("Invalid x index")
                self.x[i] ^= 1
            self._encode()
            return
        elif action == "reset":
            self.y_prime = self.truth_llr.copy()
        elif action == "noise":
            sigma = float(event["sigma"])
            if not np.isfinite(sigma) or sigma < 0:
                raise ValueError("Noise sigma must be finite and nonnegative")
            self.y_prime = self.y_prime + self.rng.normal(0, sigma, self.n)
        elif action == "paint":
            i = int(event["index"])
            index = i
            if not 0 <= i < self.n:
                raise ValueError("Invalid LLR index")
            mode, amount = event["mode"], float(event["amount"])
            if amount < 0 or np.isnan(amount):
                raise ValueError("Brush amount must be nonnegative")
            if mode == "suppress":
                self.y_prime[i] *= 10 ** (-amount / 20)
            elif mode in ("flip", "restore") and np.isfinite(amount):
                self.y_prime[i] += self.truth_llr[i] * amount * (-1 if mode == "flip" else 1)
            else:
                raise ValueError("Invalid brush mode or amount")
        else:
            raise ValueError("Unknown editor action")
        self._publish(index=index)
