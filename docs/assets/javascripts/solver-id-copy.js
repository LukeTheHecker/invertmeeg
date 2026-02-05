function copyTextFallback(text) {
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "absolute";
  textarea.style.left = "-9999px";
  document.body.appendChild(textarea);
  textarea.select();
  document.execCommand("copy");
  document.body.removeChild(textarea);
}

async function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(text);
    return;
  }
  copyTextFallback(text);
}

function attachSolverIdCopyHandlers() {
  document.querySelectorAll("[data-solver-id]").forEach((el) => {
    if (el.dataset.copyBound === "true") return;
    el.dataset.copyBound = "true";

    const originalTitle = el.getAttribute("title") || "";
    el.addEventListener("click", async (evt) => {
      evt.preventDefault();
      evt.stopPropagation();
      const solverId = el.getAttribute("data-solver-id");
      if (!solverId) return;

      try {
        await copyText(solverId);
        el.setAttribute("title", "Copied!");
        setTimeout(() => el.setAttribute("title", originalTitle), 900);
      } catch {
        // no-op
      }
    });
  });
}

document.addEventListener("DOMContentLoaded", attachSolverIdCopyHandlers);
document.addEventListener("readystatechange", attachSolverIdCopyHandlers);

