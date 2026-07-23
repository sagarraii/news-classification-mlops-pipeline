(function () {
  const textInput = document.getElementById("text-input");
  const charCount = document.getElementById("char-count");
  const transmitBtn = document.getElementById("transmit-btn");

  const outputDot = document.getElementById("output-dot");
  const outputStatus = document.getElementById("output-status");

  const idleState = document.getElementById("idle-state");
  const errorState = document.getElementById("error-state");
  const errorMessage = document.getElementById("error-message");
  const tickerState = document.getElementById("ticker-state");
  const tickerText = document.getElementById("ticker-text");
  const resultState = document.getElementById("result-state");

  const stamp = document.getElementById("stamp");
  const stampCode = document.getElementById("stamp-code");
  const stampLabel = document.getElementById("stamp-label");
  const confidenceBars = document.getElementById("confidence-bars");
  const confirmLine = document.getElementById("confirm-line");
  const latencyLine = document.getElementById("latency-line");

  if (!textInput) return; // about.html doesn't have the console

  const MAX_LEN = 2000;

  textInput.addEventListener("input", () => {
    charCount.textContent = `${textInput.value.length} / ${MAX_LEN}`;
  });

  function showState(name) {
    idleState.hidden = name !== "idle";
    errorState.hidden = name !== "error";
    tickerState.hidden = name !== "ticker";
    resultState.hidden = name !== "result";
  }

  function setStatus(text, dotClass) {
    outputStatus.textContent = text;
    outputDot.className = `dot ${dotClass}`;
  }

  function typeTicker(text, onDone) {
    tickerText.textContent = "";
    showState("ticker");
    setStatus("RECEIVING", "dot-live");
    let i = 0;
    const interval = setInterval(() => {
      tickerText.textContent += text[i];
      i += 1;
      if (i >= text.length) {
        clearInterval(interval);
        setTimeout(onDone, 220);
      }
    }, 14);
  }

  function renderConfidence(confidences) {
    confidenceBars.innerHTML = "";
    const colorMap = {
      World: "var(--world)",
      Sports: "var(--sport)",
      Business: "var(--biz)",
      "Sci/Tech": "var(--tech)",
    };
    const entries = Object.entries(confidences).sort((a, b) => b[1] - a[1]);
    entries.forEach(([label, value]) => {
      const row = document.createElement("div");
      row.className = "confidence-row";
      row.innerHTML = `
        <span>${label}</span>
        <span class="confidence-track"><span class="confidence-fill" style="background:${colorMap[label] || 'var(--wire-red)'}"></span></span>
        <span>${value.toFixed(1)}%</span>
      `;
      confidenceBars.appendChild(row);
      requestAnimationFrame(() => {
        const fill = row.querySelector(".confidence-fill");
        fill.style.width = `${value}%`;
      });
    });
  }

  async function transmit() {
    const text = textInput.value.trim();

    if (!text) {
      showState("error");
      errorMessage.textContent = "Paste a headline or article snippet first.";
      setStatus("NO SIGNAL", "dot-error");
      return;
    }

    transmitBtn.disabled = true;
    typeTicker("> establishing uplink... > cleaning text... > vectorizing... > inference... > verified ✓", async () => {
      try {
        const response = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
        });
        const data = await response.json();

        if (!response.ok) {
          showState("error");
          errorMessage.textContent = data.error || "Something went wrong on the wire.";
          setStatus("TRANSMISSION FAILED", "dot-error");
          transmitBtn.disabled = false;
          return;
        }

        stamp.style.setProperty("--stamp-color", data.color);
        stampCode.textContent = data.code;
        stampLabel.textContent = `${data.icon || ""} ${data.category}`.trim();
        stamp.classList.remove("stamp-in");
        void stamp.offsetWidth; // restart animation
        stamp.classList.add("stamp-in");

        if (typeof data.latency_ms === "number") {
          latencyLine.textContent = `Processed in ${data.latency_ms} ms`;
        } else {
          latencyLine.textContent = "";
        }

        renderConfidence(data.confidences || {});
        showState("result");
        setStatus("TRANSMISSION COMPLETE", "dot");
      } catch (err) {
        showState("error");
        errorMessage.textContent = "Couldn't reach the classification service. Is the app running?";
        setStatus("LINE DOWN", "dot-error");
      } finally {
        transmitBtn.disabled = false;
      }
    });
  }

  transmitBtn.addEventListener("click", transmit);

  textInput.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      transmit();
    }
  });
})();
