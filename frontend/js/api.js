/**
 * ResearchGPT Frontend API Client (Task 19)
 * Connects UI seamlessly to /api/v1/* RESTful backend services.
 */

const API_BASE = '/api/v1';

export class ApiClient {
  static async request(endpoint, options = {}) {
    const url = `${API_BASE}${endpoint}`;
    const headers = {
      ...(options.headers || {}),
    };

    if (!(options.body instanceof FormData) && !headers['Content-Type']) {
      headers['Content-Type'] = 'application/json';
    }

    try {
      const response = await fetch(url, {
        ...options,
        headers,
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || data.error || `HTTP ${response.status} Error`);
      }
      return data;
    } catch (err) {
      console.error(`[API Error] ${endpoint}:`, err);
      throw err;
    }
  }

  // Health
  static async getHealth() {
    return this.request('/health');
  }

  static async getDetailedHealth() {
    return this.request('/health/detailed');
  }

  // Documents
  static async listDocuments() {
    return this.request('/documents');
  }

  static async uploadPDF(file) {
    const formData = new FormData();
    formData.append('file', file);
    return this.request('/documents/upload', {
      method: 'POST',
      body: formData,
    });
  }

  static async extractPDF(documentId) {
    return this.request(`/documents/${documentId}/extract`, {
      method: 'POST',
    });
  }

  static async deleteDocument(documentId) {
    return this.request(`/documents/${documentId}`, {
      method: 'DELETE',
    });
  }

  // Preprocessing
  static async runClassicalPreprocessing(payload) {
    return this.request('/preprocessing/classical', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  static async runTransformerPreprocessing(payload) {
    return this.request('/preprocessing/transformer', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  // Classification
  static async classifyText(payload) {
    return this.request('/classification/predict', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  // NER
  static async extractEntities(payload) {
    return this.request('/ner/extract', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  // Summarization
  static async summarizeText(payload) {
    return this.request('/summarization/summarize', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  // Semantic Similarity
  static async compareSimilarity(payload) {
    return this.request('/similarity/compare', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  // Retrieval & Indexing
  static async indexDocument(payload) {
    return this.request('/retrieval/index-document', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  static async searchVectors(payload) {
    return this.request('/retrieval/search', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  static async getRetrievalStatus() {
    return this.request('/retrieval/status');
  }

  // Question Answering
  static async answerQuestion(payload) {
    return this.request('/qa/answer', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  // Grounded RAG
  static async queryRAG(payload) {
    return this.request('/rag/query', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  // Single-Pass Paper Analysis
  static async analyzePaper(payload) {
    return this.request('/analysis/paper', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }
}
