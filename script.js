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
    // 1. Captura o frame (rank 3: [height, width, 3])
    let tensor = tf.browser.fromPixels(webcamElement);
    
    // 2. Converte para grayscale (rank 2: [height, width])
    tensor = tensor.mean(2);
    
    // 3. Adiciona canal (rank 3: [height, width, 1])
    tensor = tensor.expandDims(2);
    
    // 4. Redimensiona para 96x96
    tensor = tensor.resizeNearestNeighbor([96, 96]);
    
    // 5. Normalização (comente se não for usado no treino)
    tensor = tensor.div(255.0);
    
    // 6. Adiciona dimensão de batch (rank 4: [1, 96, 96, 1])
    return tensor.expandDims(0);
  });

  try {
    // Faz a predição
    const output = model.predict(tensor);
    const predictions = await output.data();
    output.dispose();
    tensor.dispose();

    console.log("Raw predictions:", Array.from(predictions)); // DEBUG
    
    // Processa resultados
    const maxIndex = predictions.indexOf(Math.max(...predictions));
    const confidence = (predictions[maxIndex] * 100).toFixed(2);
    
    return {
      className: labels[maxIndex] || `Classe ${maxIndex}`,
      confidence: isNaN(confidence) ? "0.00" : confidence
    };
  } catch (error) {
    tensor.dispose();
    console.error("Erro na predição:", error);
    return {
      className: "Erro",
      confidence: "0.00"
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