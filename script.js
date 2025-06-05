// Variáveis globais
let model, webcamElement;
let serialPort;
let distance = 0;
let labels = ["Pedestre", "Sem Pedestre", "Carro"];
let metadata = { grayscale: true };
let isPredicting = false;
let lastAlertTime = 0;
let lineBuffer = "";
let alertTimeout = null;
let isWaitingToShowAlert = false;
let alertAudio = null; // Para controlar o áudio do alerta

// Elementos DOM
const resultElement = document.getElementById("result");
const distanceElement = document.getElementById("distance-display");
const connectButton = document.getElementById("connect-arduino");
const startButton = document.getElementById("start-classification");
const alertContainer = document.getElementById("alert-container");

// Event Listeners
connectButton.addEventListener("click", connectToArduino);
startButton.addEventListener("click", toggleClassification);

// Função principal de inicialização
async function init() {
  try {
    webcamElement = document.getElementById("webcam");
    await setupWebcam();
    await loadModel();
    startButton.disabled = false;
    resultElement.textContent = "Modelo carregado. Clique para iniciar.";
  } catch (error) {
    console.error("Erro na inicialização:", error);
    resultElement.textContent = `Erro: ${error.message}`;
  }
}

// Configura a webcam
async function setupWebcam() {
  return new Promise((resolve, reject) => {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("Webcam não suportada");
    }

    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => {
        webcamElement.srcObject = stream;
        webcamElement.addEventListener("loadeddata", resolve);
      })
      .catch(reject);
  });
}

// Carrega o modelo
async function loadModel() {
  model = await tf.loadLayersModel("model/model.json");

  try {
    const metadataResponse = await fetch("model/metadata.json");
    metadata = await metadataResponse.json();
    if (metadata.labels) labels = metadata.labels;
  } catch (e) {
    console.warn("Metadados não encontrados, usando padrão");
  }

  console.log("Modelo carregado. Configuração:", metadata);
}

// Conexão com Arduino
async function connectToArduino() {
  try {
    serialPort = await navigator.serial.requestPort();
    await serialPort.open({ baudRate: 9600 });

    connectButton.textContent = "Conectado";
    connectButton.style.backgroundColor = "#27ae60";

    const decoder = new TextDecoderStream();
    const inputDone = serialPort.readable.pipeTo(decoder.writable);
    const inputStream = decoder.readable;

    const reader = inputStream.getReader();

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      lineBuffer += value;

      let lines = lineBuffer.split("\n");
      lineBuffer = lines.pop(); // mantém linha incompleta no buffer

      for (let line of lines) {
        line = line.trim();
        const match = line.match(/Distância: ([\d.]+) cm/);
        if (match) {
          distance = parseFloat(match[1]);
          updateDistanceDisplay();
        } else {
          console.log("Dado ignorado:", line);
        }
      }
    }

  } catch (error) {
    console.error("Erro na conexão serial:", error);
    connectButton.textContent = "Erro - Tentar novamente";
    connectButton.style.backgroundColor = "#e74c3c";
  }
}

// Atualiza display da distância
function updateDistanceDisplay() {
  distanceElement.textContent = `Distância: ${distance.toFixed(1)} cm`;
}

// Classificação de imagem
async function predict() {
  const tensor = tf.tidy(() => {
    let tensor = tf.browser.fromPixels(webcamElement)
      .resizeNearestNeighbor([metadata.imageSize || 96, metadata.imageSize || 96])
      .toFloat();

    if (metadata.grayscale) {
      tensor = tensor.mean(2).expandDims(2);
    }

    tensor = tensor.div(255.0);
    return tensor.expandDims(0);
  });

  const predictions = await model.predict(tensor).data();
  tensor.dispose();

  const maxIndex = predictions.indexOf(Math.max(...predictions));
  return {
    className: labels[maxIndex] || `Classe ${maxIndex}`,
    confidence: (predictions[maxIndex] * 100).toFixed(2)
  };
}

// Loop de classificação
async function predictLoop() {
  while (isPredicting) {
    try {
      const prediction = await predict();
      resultElement.textContent =
        `${prediction.className} - ${distance.toFixed(1)}cm - Confiança: ${prediction.confidence}%`;

      checkAlertConditions(prediction);
    } catch (error) {
      console.error("Erro na predição:", error);
    }

    await tf.nextFrame();
  }
}

// Verifica condições de alerta (90%+ de confiança e distância < 20cm)
function checkAlertConditions(prediction) {
  const confidenceValue = parseFloat(prediction.confidence);
  const isPedestrian = prediction.className === "Pedestre";
  const isCloseEnough = distance < 20;

  if (isPedestrian && isCloseEnough && confidenceValue >= 90) {
    // Mostra alerta imediatamente e toca o som em loop
    alertContainer.style.display = "block";
    if (!alertAudio || alertAudio.paused) {
      alertAudio = playAlertSound();
    }
  } else {
    // Esconde o alerta e para o som
    alertContainer.style.display = "none";
    if (alertAudio) {
      alertAudio.pause();
      alertAudio.currentTime = 0;
    }
  }
}

// Toca som de alerta em loop
function playAlertSound() {
  try {
    const audio = new Audio('alert.mp3');
    audio.loop = true;
    audio.play().catch(e => console.log("Erro no áudio:", e));
    return audio;
  } catch (e) {
    console.log("Não foi possível reproduzir som:", e);
    return null;
  }
}

// Controle da classificação
function toggleClassification() {
  isPredicting = !isPredicting;

  if (isPredicting) {
    startButton.textContent = "Parar Classificação";
    startButton.style.backgroundColor = "#e74c3c";
    predictLoop();
  } else {
    startButton.textContent = "Iniciar Classificação";
    startButton.style.backgroundColor = "#2c3e50";
    resultElement.textContent = "Classificação pausada";
    alertContainer.style.display = "none";
    if (alertAudio) {
      alertAudio.pause();
      alertAudio.currentTime = 0;
    }
  }
}

// Inicia o sistema quando a página carrega
window.addEventListener('load', init);