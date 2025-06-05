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
    return tf.browser.fromPixels(webcamElement)
      .mean(2)                              // Converte para grayscale primeiro
      .resizeNearestNeighbor([96, 96])      // Redimensiona para o tamanho do modelo
      .toFloat()
      .div(255.0)                          // Normaliza [0,1] (opcional, depende do modelo)
      .expandDims(0)                        // Adiciona dimensão batch
      .expandDims(-1);                      // Adiciona canal (grayscale)
  });

  // Predição
  const predictions = await model.predict(tensor).data();
  tensor.dispose(); // Libera memória

  // Processa resultados
  const maxIndex = predictions.indexOf(Math.max(...predictions));
  const confidence = (predictions[maxIndex] * 100).toFixed(2);
  const classes = ["Pedestre", "Sem Pedestre", "Carro"]; // Do metadata.json
  
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