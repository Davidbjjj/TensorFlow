let model, webcamElement, labels = ["Pedestre", "Sem Pedestre", "Carro"]; // Fallback manual

async function setupWebcam() {
  webcamElement = document.getElementById('webcam');
  
  if (!navigator.mediaDevices?.getUserMedia) {
    alert("Seu navegador não suporta acesso à webcam ou não está em um servidor seguro (HTTPS/localhost).");
    throw new Error("getUserMedia não suportado.");
  }

  const stream = await navigator.mediaDevices.getUserMedia({ 
    video: { 
      width: 96, 
      height: 96 
    } 
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
  try {
    // 1. Carrega o modelo
    model = await tf.loadLayersModel("model/model.json");
    
    // 2. Tenta carregar metadados (opcional)
    try {
      const metadata = await fetch("model/metadata.json").then(r => r.json());
      if (metadata.labels) labels = metadata.labels;
    } catch (e) {
      console.warn("Metadados não encontrados, usando labels padrão");
    }
    
    console.log("Modelo carregado!", model);
    console.log("Labels:", labels);
  } catch (error) {
    console.error("Falha ao carregar modelo:", error);
    throw error;
  }
}

async function predict() {
  const tensor = tf.tidy(() => {
    // Versão 1: Grayscale (comente se testar a versão RGB)
    let tensor = tf.browser.fromPixels(webcamElement)
      .resizeNearestNeighbor([96, 96])
      .mean(2)
      .toFloat()
      .expandDims(0)
      .expandDims(-1);
    
    // Versão 2: RGB (descomente para testar)
    // let tensor = tf.browser.fromPixels(webcamElement)
    //   .resizeNearestNeighbor([96, 96])
    //   .toFloat()
    //   .expandDims(0);
    
    // Debug: verifique os valores
    console.log('Valores min/max:', tensor.min().dataSync()[0], tensor.max().dataSync()[0]);
    
    return tensor;
  });

  try {
    const output = model.predict(tensor);
    const predictions = await output.data();
    
    console.log('Predictions raw:', predictions);
    
    tensor.dispose();
    output.dispose();

    const maxIndex = predictions.indexOf(Math.max(...predictions));
    const confidence = (predictions[maxIndex] * 100).toFixed(2);
    
    return {
      className: labels[maxIndex],
      confidence: confidence
    };
  } catch (error) {
    tensor.dispose();
    console.error('Prediction error:', error);
    return {
      className: 'Erro',
      confidence: '0.00'
    };
  }
}
async function predictLoop() {
  while (true) {
    const result = await predict();
    document.getElementById("result").textContent = 
      `${result.className} - Confiança: ${result.confidence}%`;
    await tf.nextFrame();
  }
}

async function main() {
  try {
    await setupWebcam();
    await loadModel();
    predictLoop();
  } catch (error) {
    document.getElementById("result").textContent = 
      `Erro: ${error.message}`;
    console.error("Falha na inicialização:", error);
  }
}

// Inicia a aplicação
main();