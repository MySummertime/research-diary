// Projects/Risk_Assessment_Models/frontend/script.js

// Since we use 'defer' in the HTML, the DOM is guaranteed to be ready.
// The DOMContentLoaded event listener is no longer necessary.

// Ensure the extension is registered before creating any Cytoscape instance
cytoscape.use(cytoscapePopper);

// --- 1. Centralized Text Management (Internationalization Dictionary) ---
const translations = {
  en: {
    pageTitle: 'Risk Assessment Model Analysis Platform',
    headerTitle: 'Risk Assessment Model Analysis Platform',
    selectFileBtn: 'Select Data File',
    controlsTitle: 'Control Panel',
    sourceNodeLabel: 'Origin Node:',
    destNodeLabel: 'Destination Node:',
    alphaLabel: 'Confidence Level (α):',
    calculateBtn: 'Calculate',
    resultsTitle: 'Analysis Results',
    resultsInitial: 'Please select a data file and set parameters.',
    modalTitle: 'Select Network File',
    modalConfirmBtn: 'Confirm',
    modalCancelBtn: 'Cancel',
    fileSelected: 'Selected file: {file}. Please enter parameters.',
    loadingFile: 'Loading network file: {file}...',
    calculating: 'Calculating...',
    selectFileFirst: 'Please select a file first.',
    enterValidNumbers: 'Please enter valid numeric values for source, destination, and alpha.',
    loadFileListFail: 'Failed to load file list: {error}. Please check backend service.',
    loadGraphDataFail:
      'Failed to load graph data: {error}. Please check file format and backend service.',
    calculationSuccess:
      'Calculation Successful:\n--------------------\nOptimal Path: {path}\nConfidence Level (α): {alpha}\nBest VaR (β): {var}',
    calculationFailed: 'Calculation failed: {message}',
    errorOccurred: 'An error occurred: {error}',
  },
  zh: {
    pageTitle: '风险评估模型分析平台',
    headerTitle: '风险评估模型分析平台',
    selectFileBtn: '选择数据文件',
    controlsTitle: '控制面板',
    sourceNodeLabel: '源节点 (Origin):',
    destNodeLabel: '目标节点 (Destination):',
    alphaLabel: '置信水平 (α):',
    calculateBtn: '开始计算',
    resultsTitle: '分析结果',
    resultsInitial: '请先选择一个数据文件并设置参数。',
    modalTitle: '选择网络文件',
    modalConfirmBtn: '确认',
    modalCancelBtn: '取消',
    fileSelected: '已选择文件: {file}。请输入参数。',
    loadingFile: '正在加载网络文件: {file}...',
    calculating: '正在计算...',
    selectFileFirst: '请先选择一个文件。',
    enterValidNumbers: '请输入有效的源节点、目标节点和置信水平数值。',
    loadFileListFail: '加载文件列表失败: {error}。请检查后端服务。',
    loadGraphDataFail: '加载图数据失败: {error}。请检查文件格式和后端服务。',
    calculationSuccess:
      '计算成功:\n--------------------\n最优路径: {path}\n置信水平 (α): {alpha}\n最佳风险值 (β): {var}',
    calculationFailed: '计算失败: {message}',
    errorOccurred: '发生错误: {error}',
  },
};

// --- 2. Configuration, DOM References, and State Management ---
const CONFIG = {
  // Set relative backend URL
  backendUrl: '',
  // Set backend API endpoints
  apiEndpoints: {
    dataFiles: '/api/data_files',
    graph: '/api/graph',
    solve: '/api/solve',
  },
};

const dom = {
  languageToggle: document.getElementById('language-toggle'),
  selectFileButton: document.getElementById('select-file-button'),
  solverForm: document.getElementById('solver-form'),
  resultsOutput: document.getElementById('results-output'),
  cyContainer: document.getElementById('cy'),
  srcInput: document.getElementById('src-input'),
  destInput: document.getElementById('dest-input'),
  alphaInput: document.getElementById('alpha-input'),
  fileModal: document.getElementById('file-modal'),
  fileSelectDropdown: document.getElementById('file-select-dropdown'),
  submitFileButton: document.getElementById('submit-file-button'),
  cancelFileButton: document.getElementById('cancel-file-button'),
};

let state = {
  cyInstance: null,
  selectedFile: null,
  currentLanguage: 'en',
};

// --- 3. Core Functional Logic ---

function getString(key, replacements = {}) {
  let str = translations[state.currentLanguage]?.[key] || key;
  for (const placeholder in replacements) {
    str = str.replace(`{${placeholder}}`, replacements[placeholder]);
  }
  return str;
}

function updateContent() {
  document.querySelectorAll('[data-i18n-key]').forEach((element) => {
    const key = element.getAttribute('data-i18n-key');
    element.textContent = getString(key);
  });
  document.title = getString('pageTitle');
  if (dom.resultsOutput.getAttribute('data-i18n-base-text') === 'resultsInitial') {
    dom.resultsOutput.textContent = getString('resultsInitial');
  }
}

// Simplified function to update the results panel directly
function updateResults(text) {
  dom.resultsOutput.textContent = text;
  dom.resultsOutput.removeAttribute('data-i18n-base-text');
}

// Modal Control for file selection
function closeFileModal() {
  dom.fileModal.classList.remove('is-visible');
}

function openFileModal() {
  fetch(`${CONFIG.backendUrl}${CONFIG.apiEndpoints.dataFiles}`)
    .then((response) => {
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      return response.json();
    })
    .then((files) => {
      if (!files || files.length === 0) {
        updateResults('Network data not found in "backend/data/"');
        return;
      }
      dom.fileSelectDropdown.innerHTML = '';
      files.forEach((file) => {
        const option = document.createElement('option');
        option.value = file;
        option.textContent = file;
        dom.fileSelectDropdown.appendChild(option);
      });
      dom.fileModal.classList.add('is-visible');
    })
    .catch((error) => {
      updateResults(getString('loadFileListFail', { error: error.message }));
    });
}

// Graph-related Functions
function initializeCytoscape(elements) {
  if (state.cyInstance) {
    state.cyInstance.destroy();
  }
  state.cyInstance = cytoscape({
    container: dom.cyContainer,
    elements: elements,
    layout: { name: 'preset' },
    style: [
      {
        selector: 'node',
        style: { 'background-color': '#666', label: 'data(label)', width: 20, height: 20 },
      },
      {
        selector: 'edge',
        style: {
          width: 3,
          'line-color': '#ccc',
          'target-arrow-color': '#ccc',
          'target-arrow-shape': 'triangle',
          'curve-style': 'bezier',
          label: 'data(label)',
          'font-size': '8px',
          'text-wrap': 'wrap',
        },
      },
    ],
  });
  setupCytoscapeListeners();
}

function setupCytoscapeListeners() {
  if (!state.cyInstance) return;
  state.cyInstance.on('mouseover', 'node', (evt) => {
    const node = evt.target;
    const popperRef = node.popperRef();
    node.tip = tippy(popperRef, {
      content: `<b>${node.data('label')}</b><br>Probability: ${node.data('prob')}<br>Consequence: ${node.data('cons')}`,
      trigger: 'manual',
      allowHTML: true,
      arrow: true,
      placement: 'top',
      hideOnClick: false,
    });
    node.tip.show();
  });
  state.cyInstance.on('mouseout', 'node', (evt) => evt.target.tip?.hide());
  state.cyInstance.on('drag', 'node', (evt) => evt.target.tip?.hide());
}

function loadGraph(fileName) {
  if (!fileName) return;
  updateResults(getString('loadingFile', { file: fileName }));
  fetch(`${CONFIG.backendUrl}${CONFIG.apiEndpoints.graph}?file=${encodeURIComponent(fileName)}`)
    .then((response) => {
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      return response.json();
    })
    .then((data) => {
      const elements = [
        ...Object.values(data.nodes).map((node) => ({
          group: 'nodes',
          data: { id: String(node.id), label: String(node.id), prob: node.prob, cons: node.cons },
          position: { x: node.x, y: node.y },
        })),
        ...Object.values(data.edges).map((edge) => ({
          group: 'edges',
          data: {
            id: `e${edge.source}-${edge.target}`,
            source: String(edge.source),
            target: String(edge.target),
            label: `P=${edge.prob.toFixed(2)}\nC=${edge.cons}`,
          },
        })),
      ];
      initializeCytoscape(elements);
      updateResults(getString('fileSelected', { file: fileName }));
    })
    .catch((error) => {
      updateResults(getString('loadGraphDataFail', { error: error.message }));
    });
}

function highlightPath(path) {
  if (!state.cyInstance) return;
  state.cyInstance.elements().style({
    'background-color': '#666',
    'line-color': '#ccc',
    'target-arrow-color': '#ccc',
    width: 3,
  });
  state.cyInstance.nodes().style({ width: 20, height: 20 });
  path.forEach((nodeId, index) => {
    const node = state.cyInstance.getElementById(String(nodeId));
    if (node.length) {
      node.style({ 'background-color': 'red' });
    }
    if (index < path.length - 1) {
      const nextNodeId = path[index + 1];
      const edge = state.cyInstance.edges(`[source="${nodeId}"][target="${nextNodeId}"]`);
      if (edge.length) {
        edge.style({ 'line-color': 'red', 'target-arrow-color': 'red', width: 5 });
      }
    }
  });
}

// Form Submission Handler
function handleSolverSubmit(event) {
  event.preventDefault();
  if (!state.selectedFile) {
    updateResults(getString('selectFileFirst'));
    return;
  }
  const src = parseInt(dom.srcInput.value);
  const dest = parseInt(dom.destInput.value);
  const alpha = parseFloat(dom.alphaInput.value);
  if (isNaN(src) || isNaN(dest) || isNaN(alpha)) {
    updateResults(getString('enterValidNumbers'));
    return;
  }
  updateResults(getString('calculating'));
  const payload = { src, dest, alpha, selected_file: state.selectedFile };
  fetch(`${CONFIG.backendUrl}${CONFIG.apiEndpoints.solve}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
    .then((response) => {
      if (!response.ok) {
        return response.json().then((err) => {
          throw new Error(err.detail || `HTTP error! status: ${response.status}`);
        });
      }
      return response.json();
    })
    .then((result) => {
      if (result.status === 'success') {
        const detailedResultString = getString('calculationSuccess', {
          path: result.path.join(' -> '),
          alpha: result.alpha.toFixed(5),
          var: result.optimal_var.toFixed(2),
        });
        updateResults(detailedResultString);
        if (state.cyInstance) highlightPath(result.path);
      } else {
        updateResults(getString('calculationFailed', { message: result.message }));
      }
    })
    .catch((error) => {
      updateResults(getString('errorOccurred', { error: error.message }));
    });
}

// --- 4. Initialization and Event Listeners ---
function initialize() {
  // Language toggle
  dom.languageToggle.addEventListener('change', () => {
    state.currentLanguage = dom.languageToggle.checked ? 'zh' : 'en';
    localStorage.setItem('preferredLanguage', state.currentLanguage);
    updateContent();
  });

  // File Modal control
  dom.selectFileButton.addEventListener('click', openFileModal);
  dom.cancelFileButton.addEventListener('click', closeFileModal);
  dom.submitFileButton.addEventListener('click', () => {
    const selected = dom.fileSelectDropdown.value;
    if (selected) {
      state.selectedFile = selected;
      loadGraph(state.selectedFile);
    }
    closeFileModal();
  });
  dom.fileModal.addEventListener('click', (event) => {
    if (event.target === dom.fileModal) closeFileModal();
  });

  // Global keydown listener for the file modal
  window.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && dom.fileModal.classList.contains('is-visible')) {
      closeFileModal();
    }
  });

  // Form submission
  dom.solverForm.addEventListener('submit', handleSolverSubmit);

  // Set language on page load
  const savedLang = localStorage.getItem('preferredLanguage');
  if (savedLang) {
    state.currentLanguage = savedLang;
  } else {
    // If no saved language, set default based on browser language
    const browserLang = navigator.language.split('-')[0];
    if (browserLang === 'zh') state.currentLanguage = 'zh';
  }
  dom.languageToggle.checked = state.currentLanguage === 'zh';

  // Initial load of page content
  updateContent();
}

// Call the initialization function
initialize();
