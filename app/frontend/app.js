const preview = document.querySelector("#preview");
const cameraButton = document.querySelector("#camera-button");
const recordButton = document.querySelector("#record-button");
const fileInput = document.querySelector("#file-input");
const selection = document.querySelector("#selection");
const systemStatus = document.querySelector("#system-status");
const resultStatus = document.querySelector("#result-status");
const resultText = document.querySelector("#result-text");
const resultIntent = document.querySelector("#result-intent");
const resultGloss = document.querySelector("#result-gloss");
const resultConfidence = document.querySelector("#result-confidence");
const resultLatency = document.querySelector("#result-latency");
const warnings = document.querySelector("#warnings");

let mediaStream;

async function loadHealth() {
  try {
    const response = await fetch("/health");
    const health = await response.json();
    if (health.model_ready) {
      systemStatus.textContent = "模型已加载，可以进行真实识别。";
      systemStatus.dataset.kind = "ready";
    } else if (health.demo_mode) {
      systemStatus.textContent = "界面演示模式：结果不是模型预测，不能用于实验报告。";
      systemStatus.dataset.kind = "warning";
    } else {
      systemStatus.textContent = "模型尚未安装。可以检查界面和上传流程，但不能进行真实识别。";
      systemStatus.dataset.kind = "warning";
    }
  } catch (error) {
    systemStatus.textContent = `无法连接后端：${error.message}`;
    systemStatus.dataset.kind = "error";
  }
}

cameraButton.addEventListener("click", async () => {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    preview.srcObject = mediaStream;
    recordButton.disabled = false;
    cameraButton.textContent = "摄像头已开启";
  } catch (error) {
    selection.textContent = `无法开启摄像头：${error.message}`;
  }
});

recordButton.addEventListener("click", () => {
  if (!mediaStream) return;
  selection.textContent = "录制功能需要后端支持，请使用视频文件上传。";
  // 简化版：直接提示使用文件上传
});

fileInput.addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  preview.src = url;
  preview.style.display = "block";
  selection.textContent = `已选择：${file.name}`;
  resultStatus.textContent = "等待预测";
  resultText.textContent = "上传后点击预测...";
});

// 添加预测按钮逻辑（如果页面有预测按钮，需要添加到 HTML）
// 由于 index.html 没有预测按钮，我们使用文件选择后自动预测
fileInput.addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  
  const formData = new FormData();
  formData.append("video", file);
  
  try {
    resultStatus.textContent = "识别中...";
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    
    if (data.error) {
      warnings.textContent = `错误：${data.error}`;
      warnings.style.display = "block";
      resultStatus.textContent = "识别失败";
      return;
    }
    
    resultIntent.textContent = data.intent || "unknown";
    resultGloss.textContent = data.gloss_sequence || data.gloss || "-";
    resultText.textContent = data.text_zh || "-";
    resultConfidence.textContent = data.confidence ? (data.confidence * 100).toFixed(1) + "%" : "-";
    resultLatency.textContent = data.latency_ms?.total ? data.latency_ms.total.toFixed(1) + "ms" : "-";
    resultStatus.textContent = "识别完成";
    
    if (data.warnings && data.warnings.length > 0) {
      warnings.textContent = data.warnings.join("; ");
      warnings.style.display = "block";
    } else {
      warnings.style.display = "none";
    }
  } catch (error) {
    warnings.textContent = `请求失败：${error.message}`;
    warnings.style.display = "block";
    resultStatus.textContent = "识别失败";
  }
});

loadHealth();
