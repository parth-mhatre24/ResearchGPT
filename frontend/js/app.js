/**
 * ResearchGPT — Application Logic & View Controller (Task 19)
 */

import { ApiClient } from './api.js';

// Application State
const state = {
  currentScreen: 'dashboard',
  currentPlaygroundTab: 'cls',
  uploadedDocId: null,
  extractedPages: [],
  ragHistory: [],
  indexedDocsCount: 0,
};

// UI Helper: Toast Notifications
function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `
    <span>${type === 'success' ? '✓' : type === 'error' ? '✕' : 'ℹ'}</span>
    <div>${message}</div>
  `;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(10px)';
    setTimeout(() => toast.remove(), 250);
  }, 4000);
}

// Navigation & Screen Switcher
function switchScreen(screenId) {
  document.querySelectorAll('.screen-container').forEach((el) => {
    el.classList.remove('active');
  });

  document.querySelectorAll('.nav-item').forEach((el) => {
    el.classList.remove('active');
  });

  const targetScreen = document.getElementById(`screen-${screenId}`);
  const targetNav = document.querySelector(`.nav-item[data-screen="${screenId}"]`);

  if (targetScreen) targetScreen.classList.add('active');
  if (targetNav) targetNav.classList.add('active');

  const titleEl = document.getElementById('current-screen-title');
  if (titleEl) {
    const titles = {
      dashboard: 'System Overview & Health',
      documents: 'PDF Document Ingestion & Extraction',
      rag: 'Grounded RAG Research Assistant',
      analysis: 'Single-Pass Paper Deep-Dive Analysis',
      playground: 'Multi-Task NLP Playground',
      benchmarks: 'Empirical Benchmark Leaderboard',
    };
    titleEl.textContent = titles[screenId] || 'ResearchGPT Workspace';
  }

  state.currentScreen = screenId;
}

// Playground Tab Switcher
function switchPlaygroundTab(tabId) {
  document.querySelectorAll('.tab-pane').forEach((p) => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach((b) => b.classList.remove('active'));

  const targetPane = document.getElementById(`tab-pane-${tabId}`);
  const targetBtn = document.querySelector(`.tab-btn[data-tab="${tabId}"]`);

  if (targetPane) targetPane.classList.add('active');
  if (targetBtn) targetBtn.classList.add('active');

  state.currentPlaygroundTab = tabId;
}

// Initialize Health & Status
async function checkSystemHealth() {
  const healthPill = document.getElementById('health-status-pill');
  const ragStatusPill = document.getElementById('rag-status-pill');

  try {
    const res = await ApiClient.getDetailedHealth();
    if (res.status === 'healthy') {
      if (healthPill) {
        healthPill.innerHTML = `<span class="status-dot"></span> Backend Active (${res.environment})`;
      }
      if (ragStatusPill) {
        ragStatusPill.textContent = 'RAG Pipeline Online';
      }
    }
  } catch (err) {
    if (healthPill) {
      healthPill.innerHTML = `<span class="status-dot" style="background:#f43f5e;box-shadow:0 0 8px #f43f5e;"></span> Offline`;
    }
  }
}

// Ingestion Screen: Drag & Drop + Upload
function setupDocumentHandlers() {
  const dropzone = document.getElementById('pdf-dropzone');
  const fileInput = document.getElementById('pdf-file-input');
  const uploadBtn = document.getElementById('btn-upload-pdf');
  const uploadStatus = document.getElementById('upload-status-box');

  if (!dropzone || !fileInput) return;

  dropzone.addEventListener('click', () => fileInput.click());

  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });

  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
  });

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
      fileInput.files = e.dataTransfer.files;
      handleFileSelected(fileInput.files[0]);
    }
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
      handleFileSelected(fileInput.files[0]);
    }
  });

  function handleFileSelected(file) {
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      showToast('Please select a valid PDF file.', 'error');
      return;
    }
    const nameEl = document.getElementById('selected-file-name');
    if (nameEl) nameEl.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
    if (uploadBtn) uploadBtn.disabled = false;
  }

  if (uploadBtn) {
    uploadBtn.addEventListener('click', async () => {
      const file = fileInput.files[0];
      if (!file) return;

      uploadBtn.disabled = true;
      uploadBtn.textContent = 'Uploading...';

      try {
        const res = await ApiClient.uploadPDF(file);
        state.uploadedDocId = res.document_id;
        showToast(`PDF uploaded successfully: ${res.filename}`, 'success');

        if (uploadStatus) {
          uploadStatus.innerHTML = `
            <div style="margin-top:12px; font-size:0.82rem; color:var(--accent-emerald);">
              ✓ Stored with Document ID: <code>${res.document_id}</code> (${res.file_size_bytes} bytes)
            </div>
          `;
        }

        // Auto trigger extraction
        const extractBtn = document.getElementById('btn-extract-pdf');
        if (extractBtn) extractBtn.disabled = false;
      } catch (err) {
        showToast(`Upload failed: ${err.message}`, 'error');
      } finally {
        uploadBtn.disabled = false;
        uploadBtn.textContent = 'Upload PDF';
      }
    });
  }

  // Extraction Button
  const extractBtn = document.getElementById('btn-extract-pdf');
  if (extractBtn) {
    extractBtn.addEventListener('click', async () => {
      if (!state.uploadedDocId) return;

      extractBtn.disabled = true;
      extractBtn.textContent = 'Extracting...';

      try {
        const res = await ApiClient.extractPDF(state.uploadedDocId);
        state.extractedPages = res.pages;
        showToast(`Extracted ${res.total_pages} pages (${res.total_characters} chars)`, 'success');

        const pagesContainer = document.getElementById('extracted-pages-viewer');
        if (pagesContainer) {
          pagesContainer.innerHTML = res.pages
            .map(
              (p) => `
              <div class="glass-card" style="margin-bottom:12px; padding:16px;">
                <div style="font-weight:600; color:var(--accent-cyan); margin-bottom:6px;">Page ${p.page_number} (${p.char_count} chars)</div>
                <div style="font-family:var(--font-mono); font-size:0.8rem; max-height:160px; overflow-y:auto; white-space:pre-wrap;">${escapeHtml(p.text)}</div>
              </div>
            `
            )
            .join('');
        }

        // Enable indexing button
        const indexRagBtn = document.getElementById('btn-index-rag');
        if (indexRagBtn) indexRagBtn.disabled = false;
      } catch (err) {
        showToast(`Extraction failed: ${err.message}`, 'error');
      } finally {
        extractBtn.disabled = false;
        extractBtn.textContent = 'Extract Text Content';
      }
    });
  }

  // Index for RAG Button
  const indexRagBtn = document.getElementById('btn-index-rag');
  const ragJumpBox = document.getElementById('rag-jump-box');
  const jumpToRagBtn = document.getElementById('btn-jump-to-rag');

  if (indexRagBtn) {
    indexRagBtn.addEventListener('click', async () => {
      if (!state.uploadedDocId || !state.extractedPages.length) {
        showToast('Please extract text from the PDF first.', 'error');
        return;
      }

      indexRagBtn.disabled = true;
      indexRagBtn.textContent = 'Indexing Chunks...';

      try {
        const fullText = state.extractedPages.map((p) => p.text).join('\n\n');
        const res = await ApiClient.indexDocument({
          document_id: state.uploadedDocId,
          text: fullText,
          chunk_size: 40,
          chunk_overlap: 8,
        });

        showToast(`Indexed ${res.chunks_indexed} chunks into Vector Retrieval Database!`, 'success');
        if (uploadStatus) {
          uploadStatus.innerHTML += `
            <div style="margin-top:6px; font-size:0.82rem; color:var(--accent-cyan);">
              ✓ Indexed <code>${res.chunks_indexed}</code> chunks into 384-dim Vector Store.
            </div>
          `;
        }

        if (ragJumpBox) ragJumpBox.style.display = 'block';
        loadDocumentsList();
      } catch (err) {
        showToast(`Indexing failed: ${err.message}`, 'error');
      } finally {
        indexRagBtn.disabled = false;
        indexRagBtn.textContent = '⚡ Index for RAG';
      }
    });
  }

  // Refresh Documents List button
  const refreshDocsBtn = document.getElementById('btn-refresh-docs');
  if (refreshDocsBtn) {
    refreshDocsBtn.addEventListener('click', () => {
      loadDocumentsList();
    });
  }

  if (jumpToRagBtn) {
    jumpToRagBtn.addEventListener('click', () => {
      switchScreen('rag');
      const docFilter = document.getElementById('rag-doc-filter');
      if (docFilter && state.uploadedDocId) {
        docFilter.value = state.uploadedDocId;
        updateActiveDocBadge(state.uploadedDocId);
      }
      const chatHistory = document.getElementById('rag-chat-history');
      if (chatHistory) {
        const welcomeBubble = document.createElement('div');
        welcomeBubble.className = 'chat-bubble assistant';
        welcomeBubble.innerHTML = `
          <div>📄 <strong>Active Paper Loaded:</strong> <code>${state.uploadedDocId}</code></div>
          <div style="margin-top:6px;">This paper's text chunks are indexed in the vector store. What would you like to know about this research paper?</div>
        `;
        chatHistory.appendChild(welcomeBubble);
        chatHistory.scrollTop = chatHistory.scrollHeight;
      }
    });
  }
}

// Load and Render Uploaded Documents Table & RAG Dropdown
async function loadDocumentsList() {
  const tbody = document.getElementById('documents-table-body');
  const docFilterSelect = document.getElementById('rag-doc-filter');

  try {
    const docs = await ApiClient.listDocuments();

    // Render Table
    if (tbody) {
      if (!docs || docs.length === 0) {
        tbody.innerHTML = `<tr><td colspan="6" style="text-align:center; padding:24px; color:var(--text-muted);">No documents uploaded or indexed yet.</td></tr>`;
      } else {
        tbody.innerHTML = docs
          .map((d) => {
            const sizeStr = d.file_size_bytes ? `${(d.file_size_bytes / 1024).toFixed(1)} KB` : 'N/A';
            const dateStr = d.created_at ? d.created_at.slice(0, 19).replace('T', ' ') : 'Pre-indexed';
            const isCurrent = state.uploadedDocId === d.document_id;
            return `
              <tr style="${isCurrent ? 'background:rgba(6,182,212,0.08);' : ''}">
                <td>
                  <strong>${escapeHtml(d.original_filename || d.filename)}</strong>
                  ${isCurrent ? ' <span style="font-size:0.7rem; color:var(--accent-cyan); font-weight:600;">[ACTIVE]</span>' : ''}
                </td>
                <td><code style="font-size:0.75rem; color:var(--accent-purple);">${escapeHtml(d.document_id)}</code></td>
                <td>${sizeStr}</td>
                <td><span class="meta-pill glow">${d.chunks_indexed || 0} chunks</span></td>
                <td style="font-size:0.76rem; color:var(--text-muted);">${dateStr}</td>
                <td>
                  <div style="display:flex; gap:6px;">
                    <button class="btn btn-secondary btn-set-active-doc" data-doc-id="${escapeHtml(d.document_id)}" style="padding:4px 8px; font-size:0.72rem; border-color:rgba(6,182,212,0.3); color:var(--accent-cyan);">
                      💬 Chat
                    </button>
                    <button class="btn btn-secondary btn-delete-doc" data-doc-id="${escapeHtml(d.document_id)}" data-doc-name="${escapeHtml(d.original_filename || d.filename)}" style="padding:4px 8px; font-size:0.72rem; border-color:rgba(244,63,94,0.3); color:var(--accent-rose);">
                      🗑️ Delete
                    </button>
                  </div>
                </td>
              </tr>
            `;
          })
          .join('');

        // Wire Table Action Buttons
        tbody.querySelectorAll('.btn-set-active-doc').forEach((btn) => {
          btn.addEventListener('click', () => {
            const docId = btn.getAttribute('data-doc-id');
            state.uploadedDocId = docId;
            switchScreen('rag');
            if (docFilterSelect) {
              docFilterSelect.value = docId;
              updateActiveDocBadge(docId);
            }
            showToast(`Active paper set to: ${docId}`, 'info');
          });
        });

        tbody.querySelectorAll('.btn-delete-doc').forEach((btn) => {
          btn.addEventListener('click', async () => {
            const docId = btn.getAttribute('data-doc-id');
            const docName = btn.getAttribute('data-doc-name');
            if (confirm(`Are you sure you want to delete "${docName}" and purge all its vectors from memory?`)) {
              await deleteDocumentHandler(docId);
            }
          });
        });
      }
    }

    // Populate RAG Dropdown
    if (docFilterSelect) {
      const currentVal = docFilterSelect.value;
      let optHtml = `<option value="">All Indexed Papers (Global Knowledge)</option>`;
      docs.forEach((d) => {
        const label = `${d.original_filename || d.filename} (${d.chunks_indexed || 0} chunks)`;
        optHtml += `<option value="${escapeHtml(d.document_id)}">${escapeHtml(label)}</option>`;
      });
      docFilterSelect.innerHTML = optHtml;
      if (currentVal && docs.some((d) => d.document_id === currentVal)) {
        docFilterSelect.value = currentVal;
      } else if (state.uploadedDocId && docs.some((d) => d.document_id === state.uploadedDocId)) {
        docFilterSelect.value = state.uploadedDocId;
      }
      updateActiveDocBadge(docFilterSelect.value);
    }
  } catch (err) {
    console.error('Failed to load documents list:', err);
    if (tbody) {
      tbody.innerHTML = `<tr><td colspan="6" style="text-align:center; padding:20px; color:var(--accent-rose);">Failed to load documents: ${escapeHtml(err.message)}</td></tr>`;
    }
  }
}

// Delete Document Handler
async function deleteDocumentHandler(documentId) {
  try {
    const res = await ApiClient.deleteDocument(documentId);
    showToast(res.message || `Deleted document ${documentId}`, 'success');

    if (state.uploadedDocId === documentId) {
      state.uploadedDocId = null;
    }

    // Reload documents list and dropdown
    await loadDocumentsList();
  } catch (err) {
    showToast(`Failed to delete document: ${err.message}`, 'error');
  }
}

function updateActiveDocBadge(docId) {
  const badge = document.getElementById('rag-active-doc-badge');
  if (!badge) return;
  if (docId) {
    badge.textContent = `Active Scope: ${docId.slice(0, 18)}... (Restricted to single paper)`;
    badge.style.color = 'var(--accent-emerald)';
  } else {
    badge.textContent = 'Active Scope: Global (Searches across all indexed research papers)';
    badge.style.color = 'var(--accent-cyan)';
  }
}

// Grounded RAG Chat Controller
function setupRAGChatHandlers() {
  const form = document.getElementById('rag-chat-form');
  const input = document.getElementById('rag-chat-input');
  const chatHistory = document.getElementById('rag-chat-history');
  const topKSelect = document.getElementById('rag-top-k');
  const simSlider = document.getElementById('rag-sim-threshold');
  const simValLabel = document.getElementById('rag-sim-val');
  const docFilterSelect = document.getElementById('rag-doc-filter');

  if (simSlider && simValLabel) {
    simSlider.addEventListener('input', () => {
      simValLabel.textContent = simSlider.value;
    });
  }

  if (docFilterSelect) {
    docFilterSelect.addEventListener('change', () => {
      state.uploadedDocId = docFilterSelect.value || null;
      updateActiveDocBadge(docFilterSelect.value);
    });
  }

  if (form && input && chatHistory) {
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const query = input.value.trim();
      if (!query) return;

      // Append user bubble
      appendMessage('user', query);
      input.value = '';

      // Append loading bubble
      const loadingBubble = appendMessage('assistant', 'Searching indexed documents and generating grounded answer...');

      try {
        const topK = parseInt(topKSelect ? topKSelect.value : '3', 10);
        const threshold = parseFloat(simSlider ? simSlider.value : '0.25');
        const selectedDoc = docFilterSelect ? docFilterSelect.value : (state.uploadedDocId || undefined);

        const res = await ApiClient.queryRAG({
          query,
          top_k: topK,
          similarity_threshold: threshold,
          document_id: selectedDoc || undefined,
          generation_mode: 'generative',
        });

        // Update assistant bubble
        let contentHtml = `<div style="line-height:1.6; font-size:0.92rem;">${escapeHtml(res.answer)}</div>`;

        // Grounding status tag
        const confVal = (res.confidence_score !== undefined ? res.confidence_score : res.confidence) || 0;
        if (res.is_grounded) {
          contentHtml += `<div style="margin-top:8px; font-size:0.75rem; color:var(--accent-emerald);">✓ Grounded Answer (Confidence: ${(confVal * 100).toFixed(1)}% | ${res.total_latency_ms.toFixed(1)}ms)</div>`;
        } else {
          contentHtml += `<div style="margin-top:8px; font-size:0.75rem; color:var(--accent-rose);">⚠ Insufficient Evidence (Refusal Guard Activated)</div>`;
        }

        // Collapsible Citations Accordion (Hidden by default under arrow button)
        if (res.citations && res.citations.length > 0) {
          contentHtml += `
            <div class="citation-accordion">
              <button type="button" class="citation-toggle-btn" onclick="this.parentElement.classList.toggle('open')">
                <span class="citation-arrow">▶</span>
                <span>Sources & Citations (${res.citations.length})</span>
              </button>
              <div class="citation-drawer">
          `;
          res.citations.forEach((c) => {
            const snippet = c.text_snippet || c.text_preview || '';
            contentHtml += `
              <div class="citation-card">
                <div class="citation-header">
                  <span><strong>[${c.citation_index || '#'}]</strong> ${escapeHtml(c.document_id || 'Document')} • Chunk: ${escapeHtml(c.chunk_id)}</span>
                  <span class="citation-badge">Sim: ${(c.similarity_score !== undefined ? c.similarity_score.toFixed(4) : '')}</span>
                </div>
                ${c.section_title ? `<div style="font-size:0.72rem; color:var(--accent-purple); margin-bottom:4px;">Section: ${escapeHtml(c.section_title)} ${c.page_number ? `(Page ${c.page_number})` : ''}</div>` : ''}
                <div class="citation-snippet">"${escapeHtml(snippet)}"</div>
              </div>
            `;
          });
          contentHtml += `
              </div>
            </div>
          `;
        }

        loadingBubble.innerHTML = contentHtml;
      } catch (err) {
        loadingBubble.innerHTML = `<div style="color:var(--accent-rose);">Error: ${escapeHtml(err.message)}</div>`;
      }
    });
  }

  function appendMessage(role, text) {
    const bubble = document.createElement('div');
    bubble.className = `chat-bubble ${role}`;
    bubble.textContent = text;
    chatHistory.appendChild(bubble);
    chatHistory.scrollTop = chatHistory.scrollHeight;
    return bubble;
  }
}

// Single-Pass Paper Analysis Handler
function setupPaperAnalysisHandlers() {
  const form = document.getElementById('paper-analysis-form');
  const docIdInput = document.getElementById('analysis-doc-id');
  const textInput = document.getElementById('analysis-paper-text');
  const autoIndexCheckbox = document.getElementById('analysis-auto-index');
  const resultsCard = document.getElementById('analysis-results-card');
  const submitBtn = document.getElementById('btn-run-analysis');

  if (!form || !submitBtn) return;

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const docId = docIdInput.value.trim() || 'paper_' + Date.now();
    const text = textInput.value.trim();

    if (!text) {
      showToast('Please provide paper text to analyze.', 'error');
      return;
    }

    submitBtn.disabled = true;
    submitBtn.textContent = 'Analyzing Paper...';

    try {
      const res = await ApiClient.analyzePaper({
        document_id: docId,
        text: text,
        auto_index: autoIndexCheckbox.checked,
        max_summary_length: 120,
      });

      showToast(`Paper analysis complete in ${res.analysis_latency_ms.toFixed(1)} ms!`, 'success');

      if (resultsCard) {
        resultsCard.style.display = 'block';

        // Render Summary
        const summaryEl = document.getElementById('analysis-summary-content');
        if (summaryEl) {
          summaryEl.innerHTML = `
            <p style="font-size:0.92rem; line-height:1.6; color:#ffffff;">${escapeHtml(res.summary.summary)}</p>
            <div style="margin-top:8px; font-size:0.78rem; color:var(--text-secondary);">
              Compression: ${(res.summary.compression_ratio * 100).toFixed(1)}% | ${res.summary.summary_length_words} words (from ${res.summary.input_length_words} words)
            </div>
          `;
        }

        // Render Entities
        const entitiesEl = document.getElementById('analysis-entities-content');
        if (entitiesEl) {
          if (res.entities && res.entities.length > 0) {
            entitiesEl.innerHTML = res.entities
              .map(
                (ent) => `
                <span class="entity-tag ${ent.label}">
                  ${escapeHtml(ent.text)} <small style="opacity:0.75;">${ent.label}</small>
                </span>
              `
              )
              .join(' ');
          } else {
            entitiesEl.innerHTML = '<span style="color:var(--text-muted);">No technical entities recognized.</span>';
          }
        }

        // Render Index confirmation
        const indexEl = document.getElementById('analysis-index-content');
        if (indexEl) {
          indexEl.innerHTML = `
            <span style="color:var(--accent-emerald); font-weight:600;">✓ ${res.chunks_indexed} chunks successfully indexed into Vector Store</span>
          `;
        }
      }
    } catch (err) {
      showToast(`Analysis failed: ${err.message}`, 'error');
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = 'Run Deep-Dive Analysis';
    }
  });
}

// Multi-Task NLP Playground Handlers
function setupPlaygroundHandlers() {
  // 1. Classification
  const clsBtn = document.getElementById('btn-run-cls');
  if (clsBtn) {
    clsBtn.addEventListener('click', async () => {
      const text = document.getElementById('cls-input-text').value.trim();
      const model = document.getElementById('cls-model-select').value;
      if (!text) return showToast('Please enter text to classify.', 'error');

      clsBtn.disabled = true;
      try {
        const res = await ApiClient.classifyText({ text, model_type: model });
        const out = document.getElementById('cls-result-box');
        if (out) {
          out.innerHTML = `
            <div style="font-size:1.1rem; font-weight:700; color:var(--accent-emerald); margin-bottom:8px;">
              Predicted Label: <code>${res.predicted_label}</code> (Confidence: ${((res.confidence || 0) * 100).toFixed(1)}%)
            </div>
            ${
              res.probabilities
                ? Object.entries(res.probabilities)
                    .map(
                      ([lbl, prob]) => `
                  <div class="prob-bar-container">
                    <div class="prob-label">${lbl}</div>
                    <div class="prob-track"><div class="prob-fill" style="width:${prob * 100}%"></div></div>
                    <div class="prob-val">${(prob * 100).toFixed(1)}%</div>
                  </div>
                `
                    )
                    .join('')
                : ''
            }
          `;
        }
      } catch (err) {
        showToast(err.message, 'error');
      } finally {
        clsBtn.disabled = false;
      }
    });
  }

  // 2. NER
  const nerBtn = document.getElementById('btn-run-ner');
  if (nerBtn) {
    nerBtn.addEventListener('click', async () => {
      const text = document.getElementById('ner-input-text').value.trim();
      const method = document.getElementById('ner-method-select').value;
      if (!text) return showToast('Please enter text for entity recognition.', 'error');

      nerBtn.disabled = true;
      try {
        const res = await ApiClient.extractEntities({ text, method });
        const out = document.getElementById('ner-result-box');
        if (out) {
          out.innerHTML = `
            <div style="margin-bottom:12px;">
              ${
                res.entities.length > 0
                  ? res.entities
                      .map(
                        (e) => `
                    <span class="entity-tag ${e.label}">
                      ${escapeHtml(e.text)} <small>${e.label}</small>
                    </span>
                  `
                      )
                      .join(' ')
                  : '<span style="color:var(--text-muted);">No entities extracted.</span>'
              }
            </div>
            <div style="font-size:0.75rem; color:var(--text-muted);">Model: ${res.model_type} | Tokens: ${res.tokens.length} | Latency: ${res.latency_ms || 0}ms</div>
          `;
        }
      } catch (err) {
        showToast(err.message, 'error');
      } finally {
        nerBtn.disabled = false;
      }
    });
  }

  // 3. Summarization
  const sumBtn = document.getElementById('btn-run-sum');
  if (sumBtn) {
    sumBtn.addEventListener('click', async () => {
      const text = document.getElementById('sum-input-text').value.trim();
      const maxLen = parseInt(document.getElementById('sum-max-len').value || '100', 10);
      if (!text) return showToast('Please enter text to summarize.', 'error');

      sumBtn.disabled = true;
      try {
        const res = await ApiClient.summarizeText({ text, max_length: maxLen, min_length: 20 });
        const out = document.getElementById('sum-result-box');
        if (out) {
          out.innerHTML = `
            <div style="font-size:0.92rem; color:#ffffff; line-height:1.6; margin-bottom:8px;">${escapeHtml(res.summary)}</div>
            <div style="font-size:0.78rem; color:var(--accent-cyan);">
              Compression: ${(res.compression_ratio * 100).toFixed(1)}% | Generated in ${res.execution_time_sec.toFixed(2)}s
            </div>
          `;
        }
      } catch (err) {
        showToast(err.message, 'error');
      } finally {
        sumBtn.disabled = false;
      }
    });
  }

  // 4. Semantic Similarity
  const simBtn = document.getElementById('btn-run-sim');
  if (simBtn) {
    simBtn.addEventListener('click', async () => {
      const textA = document.getElementById('sim-text-a').value.trim();
      const textB = document.getElementById('sim-text-b').value.trim();
      const method = document.getElementById('sim-method-select').value;
      if (!textA || !textB) return showToast('Please enter both sentences to compare.', 'error');

      simBtn.disabled = true;
      try {
        const res = await ApiClient.compareSimilarity({ text_a: textA, text_b: textB, method });
        const out = document.getElementById('sim-result-box');
        if (out) {
          out.innerHTML = `
            <div style="font-size:1.2rem; font-weight:700; color:var(--accent-cyan); margin-bottom:4px;">
              Cosine Similarity: ${(res.similarity_score * 100).toFixed(1)}%
            </div>
            <div style="font-size:0.85rem; color:var(--text-secondary);">
              STS-B Scale: <strong>${res.score_stsb_scale.toFixed(2)} / 5.00</strong> (${res.method})
            </div>
          `;
        }
      } catch (err) {
        showToast(err.message, 'error');
      } finally {
        simBtn.disabled = false;
      }
    });
  }

  // 5. Extractive QA
  const qaBtn = document.getElementById('btn-run-qa');
  if (qaBtn) {
    qaBtn.addEventListener('click', async () => {
      const question = document.getElementById('qa-question-input').value.trim();
      const context = document.getElementById('qa-context-input').value.trim();
      if (!question || !context) return showToast('Please provide both a question and context passage.', 'error');

      qaBtn.disabled = true;
      try {
        const res = await ApiClient.answerQuestion({ question, context });
        const out = document.getElementById('qa-result-box');
        if (out) {
          out.innerHTML = `
            <div style="font-size:1.1rem; font-weight:700; color:var(--accent-emerald); margin-bottom:6px;">
              Extracted Answer: <code>${escapeHtml(res.answer)}</code>
            </div>
            <div style="font-size:0.78rem; color:var(--text-secondary);">
              Confidence: ${(res.confidence_score * 100).toFixed(1)}% | Span: [${res.start_char}, ${res.end_char}] | Latency: ${res.latency_ms.toFixed(1)}ms
            </div>
          `;
        }
      } catch (err) {
        showToast(err.message, 'error');
      } finally {
        qaBtn.disabled = false;
      }
    });
  }
}

// Utility: HTML Sanitizer
function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

// Global DOM Loaded Entrypoint
document.addEventListener('DOMContentLoaded', () => {
  // Navigation
  document.querySelectorAll('.nav-item').forEach((btn) => {
    btn.addEventListener('click', () => {
      const screenId = btn.getAttribute('data-screen');
      if (screenId) switchScreen(screenId);
    });
  });

  // Playground sub-tabs
  document.querySelectorAll('.tab-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const tabId = btn.getAttribute('data-tab');
      if (tabId) switchPlaygroundTab(tabId);
    });
  });

  // Quick Action Buttons on Dashboard
  document.querySelectorAll('[data-goto]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const target = btn.getAttribute('data-goto');
      if (target) switchScreen(target);
    });
  });

  // Setup Screen Controllers
  checkSystemHealth();
  setupDocumentHandlers();
  loadDocumentsList();
  setupRAGChatHandlers();
  setupPaperAnalysisHandlers();
  setupPlaygroundHandlers();
});
