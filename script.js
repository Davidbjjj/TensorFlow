let model, webcamElement;

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
  model = await tf.loadLayersModel("model/model.json");
  console.log("Modelo carregado com sucesso!");
}

async function predictLoop() {
  while (true) {
    const { className, confidence } = await predict();
    document.getElementById("result").innerHTML = 
      `${className} - Confiança: ${confidence}%`;
    await tf.nextFrame();
  }
}

async function predict() {
  // Captura e pré-processamento
  const tensor = tf.tidy(() => {
    // 1. Captura da webcam (já é rank 3: [height, width, 3] - RGB)
    let tensor = tf.browser.fromPixels(webcamElement);
    
    // 2. Converte para escala de cinza (resulta em rank 2: [height, width])
    tensor = tensor.mean(2);
    
    // 3. Adiciona dimensão de canal para ficar rank 3: [height, width, 1]
    tensor = tensor.expandDims(2);
    
    // 4. Redimensiona para o tamanho esperado pelo modelo (96x96)
    tensor = tensor.resizeNearestNeighbor([96, 96]);
    
    // 5. Converte para float e normaliza se necessário
    tensor = tensor.toFloat().div(255.0);
    
    // 6. Adiciona dimensão de batch para ficar rank 4: [1, height, width, 1]
    return tensor.expandDims(0);
  });

  // Predição
  const predictions = await model.predict(tensor).data();
  tensor.dispose();

  // Processa resultados
  const maxIndex = predictions.indexOf(Math.max(...predictions));
  const confidence = (predictions[maxIndex] * 100).toFixed(2);
  const classes = ["Pedestre", "Sem Pedestre", "Carro"];
  
  return {
    className: classes[maxIndex],
    confidence: confidence
  };
}

async function main() {
  try {
    await setupWebcam();
    await loadModel();
    predictLoop();
  } catch (error) {
    console.error("Erro:", error);
    alert(`Erro: ${error.message}`);
  }
}

main();