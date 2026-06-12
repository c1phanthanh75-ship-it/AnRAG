const docsEl = document.querySelector("#documents");
const statusEl = document.querySelector("#status");
const answerEl = document.querySelector("#answer");
const traceEl = document.querySelector("#trace");
const contextsEl = document.querySelector("#contexts");
const reasoningEl = document.querySelector("#reasoning");

function setStatus(message) {
  statusEl.textContent = message || "";
}

async function refreshDocuments() {
  const response = await fetch("/api/documents");
  const data = await response.json();
  docsEl.innerHTML = "";
  if (!data.documents.length) {
    docsEl.innerHTML = '<div class="doc"><span>No PDFs ingested yet.</span></div>';
    return;
  }
  for (const doc of data.documents) {
    const item = document.createElement("label");
    item.className = "doc doc-selectable";
    item.innerHTML = `
      <input type="checkbox" class="doc-check" value="${doc.id}" />
      <div class="doc-info">
        <strong>${doc.name}</strong>
        <span>${doc.chunk_count} chunks · ${doc.anchor_count} anchors</span>
      </div>
    `;
    docsEl.appendChild(item);
  }
}

document.querySelector("#refreshDocs").addEventListener("click", refreshDocuments);

document.querySelector("#uploadForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = document.querySelector("#pdfInput");
  if (!input.files.length) {
    setStatus("Choose a PDF first.");
    return;
  }
  const data = new FormData();
  data.append("file", input.files[0]);
  setStatus("Ingesting PDF and rebuilding indexes...");
  const response = await fetch("/api/ingest", { method: "POST", body: data });
  if (!response.ok) {
    setStatus(`Upload failed: ${response.status}`);
    return;
  }
  await refreshDocuments();
  setStatus("Ready.");
});

document.querySelector("#queryForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const checkedBoxes = document.querySelectorAll(".doc-check:checked");
  const selectedDocIds = Array.from(checkedBoxes).map(cb => cb.value);
  const payload = {
    query: document.querySelector("#queryInput").value,
    mode: document.querySelector("#modeInput").value,
    budget_tokens: Number(document.querySelector("#budgetInput").value),
    generate_answer: document.querySelector("#answerInput").checked,
    doc_ids: selectedDocIds.length ? selectedDocIds : null,
  };
  if (!payload.query.trim()) {
    setStatus("Enter a query.");
    return;
  }
  setStatus("Retrieving...");
  answerEl.textContent = "";
  traceEl.innerHTML = "";
  contextsEl.innerHTML = "";

  const startTime = performance.now();
  
  const response = await fetch("/api/query_stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    setStatus(`Query failed: ${response.status}`);
    return;
  }
  
  reasoningEl.style.display = "none";
  reasoningEl.textContent = "";
  let fullAnswer = "";

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    
    const lines = buffer.split('\n\n');
    buffer = lines.pop(); // keep incomplete chunk
    
    for (const chunk of lines) {
      if (!chunk.trim()) continue;
      
      const eventMatch = chunk.match(/^event: (.*)$/m);
      const dataMatch = chunk.match(/^data: (.*)$/m);
      
      if (!eventMatch || !dataMatch) continue;
      
      const eventName = eventMatch[1];
      const dataStr = dataMatch[1];
      const data = JSON.parse(dataStr);
      
      if (eventName === "meta") {
          traceEl.innerHTML = Object.entries(data.trace)
            .map(([key, val]) => `<div><strong>${key}</strong>: ${val}</div>`)
            .join("");
          contextsEl.innerHTML = data.contexts
            .map((ctx, index) => {
              const path = ctx.path && ctx.path.length ? ctx.path.join(" > ") : "root";
              const anchor = ctx.anchor_type ? `<span class="badge">${ctx.anchor_type}</span>` : "";
              const role = ctx.chunk_role ? `<span class="badge">${ctx.chunk_role}</span>` : "";
              const ocr = ctx.ocr_applied ? `<span>OCR ${ctx.ocr_confidence ? Number(ctx.ocr_confidence).toFixed(2) : "on"}</span>` : "";
              const visual = ctx.visual_url ? `<img class="visual" src="${ctx.visual_url}" alt="PDF visual crop" />` : "";
              return `
                <article class="context">
                  <header>
                    <span>#${index + 1}</span>
                    <span>page ${ctx.page_start}-${ctx.page_end}</span>
                    <span>${ctx.tokens} tokens</span>
                    ${anchor}
                    ${role}
                    ${ocr}
                    <span>${path}</span>
                  </header>
                  ${visual}
                  <p>${ctx.text}</p>
                </article>
              `;
            })
            .join("");
      } else if (eventName === "chunk") {
          fullAnswer += data;
          
          const thinkMatch = fullAnswer.match(/<think>([\s\S]*?)<\/think>/);
          if (thinkMatch) {
              reasoningEl.style.display = "block";
              reasoningEl.textContent = thinkMatch[1].trim();
              answerEl.textContent = fullAnswer.replace(/<think>[\s\S]*?<\/think>/, "").trim();
          } else if (fullAnswer.includes("<think>")) {
              const parts = fullAnswer.split("<think>");
              reasoningEl.style.display = "block";
              reasoningEl.textContent = parts[1].trim();
              answerEl.textContent = parts[0].trim();
          } else {
              answerEl.textContent = fullAnswer;
          }
      } else if (eventName === "done") {
          const totalMs = Math.round(performance.now() - startTime);
          setStatus(`Done in ${totalMs / 1000}s`);
      }
    }
  }
});

refreshDocuments();
