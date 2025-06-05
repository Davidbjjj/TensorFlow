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
    // 1. Captura o frame e converte para float32
    let tensor = tf.browser.fromPixels(webcamElement)
      .resizeNearestNeighbor([96, 96])
      .toFloat();
    
    // 2. Converter para grayscale SE o modelo foi treinado assim
    if (metadata && metadata.grayscale) {
      tensor = tensor.mean(2).expandDims(2);
    }
    
    // 3. Normalização CRÍTICA (experimente ambas as versões)
    // Versão A: Normalização padrão (para modelos treinados com pixels [0,1])
    tensor = tensor.div(255.0);
    
    // Versão B: Normalização alternativa (se a Versão A não funcionar)
    // tensor = tensor.sub(128).div(128);  // Para modelos que esperam [-1,1]
    
    // 4. Adiciona dimensão de batch
    return tensor.expandDims(0);
  });

  // DEBUG: Verifique os valores finais
  const [min, max] = await Promise.all([tensor.min().data(), tensor.max().data()]);
  console.log('Tensor final - min:', min[0], 'max:', max[0], 'shape:', tensor.shape);

  try {
    const predictions = await model.predict(tensor).data();
    tensor.dispose();
    
    console.log('Predições:', Array.from(predictions));
    
    // Se ainda der NaN, force valores válidos para debug
    if (predictions.some(isNaN)) {
      console.warn('NaN detectado, substituindo por valores de debug');
      return {
        className: labels[0] || 'Debug',
        confidence: '100.00'
      };
    }
    
    const maxIndex = predictions.indexOf(Math.max(...predictions));
    return {
      className: labels[maxIndex],
      confidence: (predictions[maxIndex] * 100).toFixed(2)
    };
  } catch (error) {
    tensor.dispose();
    console.error('Erro na predição:', error);
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