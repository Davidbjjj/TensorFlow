let model, webcamElement, labels = ["Pedestre", "Sem Pedestre", "Carro"];
let metadata = { grayscale: true }; // Valor padrão

async function init() {
  try {
    // 1. Configura webcam
    webcamElement = document.getElementById('webcam');
    await setupWebcam();
    
    // 2. Carrega modelo e metadados
    await loadModel();
    
    // 3. Inicia loop de predição
    predictLoop();
    
    document.getElementById("result").textContent = "Modelo pronto!";
  } catch (error) {
    document.getElementById("result").textContent = `Erro: ${error.message}`;
    console.error("Falha na inicialização:", error);
  }
}

async function setupWebcam() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("Webcam não suportada");
  }
  
  const stream = await navigator.mediaDevices.getUserMedia({ 
    video: { width: 96, height: 96 } 
  });
  webcamElement.srcObject = stream;
  
  return new Promise((resolve) => {
    webcamElement.onloadedmetadata = () => {
      webcamElement.width = 96;
      webcamElement.height = 96;
      resolve();
    };
  });
}

async function loadModel() {
  // 1. Carrega modelo
  model = await tf.loadLayersModel("model/model.json");
  
  // 2. Tenta carregar metadados
  try {
    metadata = await fetch("model/metadata.json").then(r => r.json());
    if (metadata.labels) labels = metadata.labels;
  } catch (e) {
    console.warn("Metadados não encontrados, usando configuração padrão");
  }
  
  console.log("Modelo carregado. Configuração:", {
    grayscale: metadata.grayscale,
    labels: labels
  });
}

async function predict() {
  const tensor = tf.tidy(() => {
    // 1. Captura frame (96x96)
    let tensor = tf.browser.fromPixels(webcamElement)
      .resizeNearestNeighbor([96, 96])
      .toFloat();
    
    // 2. Converte para grayscale se necessário
    if (metadata.grayscale) {
      tensor = tensor.mean(2).expandDims(2);
    }
    
    // 3. Normalização (TESTE AMBAS OPÇÕES)
    // Opção A (comente uma delas):
    tensor = tensor.div(255.0); // Para modelos [0,1]
    // Opção B:
    // tensor = tensor.sub(127.5).div(127.5); // Para modelos [-1,1]
    
    return tensor.expandDims(0);
  });

  try {
    const predictions = await model.predict(tensor).data();
    tensor.dispose();
    
    // Debug final
    console.log("Predições:", Array.from(predictions));
    
    if (predictions.some(isNaN)) {
      throw new Error("Predições inválidas (NaN)");
    }
    
    const maxIndex = predictions.indexOf(Math.max(...predictions));
    return {
      className: labels[maxIndex],
      confidence: (predictions[maxIndex] * 100).toFixed(2)
    };
  } catch (error) {
    tensor.dispose();
    throw error;
  }
}

async function predictLoop() {
  while (true) {
    try {
      const result = await predict();
      document.getElementById("result").textContent = 
        `${result.className} - ${result.confidence}%`;
    } catch (error) {
      console.error("Erro na predição:", error);
      document.getElementById("result").textContent = 
        "Erro na predição - verifique o console";
    }
    await tf.nextFrame();
  }
}

// Inicia tudo quando a página carregar
window.addEventListener('load', init);