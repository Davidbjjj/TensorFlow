let model, webcamElement, labels;

async function setupWebcam() {
  webcamElement = document.getElementById('webcam');
  
  if (!navigator.mediaDevices?.getUserMedia) {
    alert("Seu navegador não suporta acesso à webcam ou não está em um servidor seguro (HTTPS/localhost).");
    throw new Error("getUserMedia não suportado.");
  }

  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  webcamElement.srcObject = stream;
  
  return new Promise((resolve) => {
    webcamElement.onloadedmetadata = resolve;
  });
}

async function loadModel() {
  // Carrega o modelo e os metadados
  model = await tf.loadLayersModel("model/model.json");
  
  // Carrega as labels do arquivo metadata.json
  const metadata = await fetch("model/metadata.json").then(response => response.json());
  labels = metadata.labels;
  
  console.log("Modelo carregado com sucesso!");
  console.log("Labels disponíveis:", labels);
}

async function predictLoop() {
  try {
    while (true) {
      const prediction = await predict();
      document.getElementById("result").innerHTML = 
        `${prediction.className} - Confiança: ${prediction.confidence}%`;
      await tf.nextFrame();
    }
  } catch (error) {
    console.error("Erro no predictLoop:", error);
  }
}

async function predict() {
  const tensor = tf.tidy(() => {
    // 1. Captura do frame (rank 3: [height, width, 3])
    let tensor = tf.browser.fromPixels(webcamElement);
    
    // 2. Converte para grayscale (rank 2: [height, width])
    tensor = tensor.mean(2);
    
    // 3. Adiciona canal (rank 3: [height, width, 1])
    tensor = tensor.expandDims(2);
    
    // 4. Redimensiona para 96x96
    tensor = tensor.resizeNearestNeighbor([96, 96]);
    
    // 5. Normalização (verifique se seu modelo foi treinado com dados normalizados)
    tensor = tensor.div(255.0);
    
    // 6. Adiciona dimensão de batch (rank 4: [1, height, width, 1])
    return tensor.expandDims(0);
  });

  try {
    const predictions = await model.predict(tensor).data();
    tensor.dispose();

    // Encontra a classe com maior probabilidade
    const maxIndex = predictions.indexOf(Math.max(...predictions));
    const confidence = (predictions[maxIndex] * 100).toFixed(2);

    return {
      className: labels ? labels[maxIndex] : `Classe ${maxIndex}`,
      confidence: confidence
    };
  } catch (error) {
    tensor.dispose();
    throw error;
  }
}

async function main() {
  try {
    await setupWebcam();
    await loadModel();
    predictLoop();
  } catch (error) {
    console.error("Erro na inicialização:", error);
    alert(`Erro: ${error.message}`);
  }
}

// Inicia a aplicação
main();