let model, webcamElement;

async function setupWebcam() {
  webcamElement = document.getElementById('webcam');

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    alert("Seu navegador não suporta acesso à webcam ou não está em um servidor seguro.");
    throw new Error("getUserMedia não suportado.");
  }

  return new Promise((resolve, reject) => {
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => {
        webcamElement.srcObject = stream;
        webcamElement.addEventListener("loadeddata", resolve);
      })
      .catch(err => {
        alert("Erro ao acessar a webcam: " + err);
        reject(err);
      });
  });
}


async function loadModel() {
  model = await tf.loadLayersModel("model/model.json");
  console.log("Modelo carregado!");
}

async function predictLoop() {
  while (true) {
    const prediction = await predict();
    document.getElementById("result").innerHTML = prediction;
    await tf.nextFrame(); // aguarda próximo frame de vídeo
  }
}

async function predict() {
  const prediction = tf.tidy(() => {
    const tensor = tf.browser.fromPixels(webcamElement)
      .resizeNearestNeighbor([96, 96])
      .mean(2)
      .toFloat()
      .expandDims(2) // de [96,96] para [96,96,1]
      .expandDims(0); // de [96,96,1] para [1,96,96,1]

    return model.predict(tensor); // isso retorna um tensor
  });

  if (!prediction) {
    return 'Erro: modelo não retornou nenhuma previsão';
  }

  const predictions = await prediction.data(); // Float32Array
  prediction.dispose();

  if (!predictions || predictions.length === 0 || isNaN(predictions[0])) {
    return 'Erro: previsão inválida (NaN ou vazio)';
  }

  const maxIndex = predictions.indexOf(Math.max(...predictions));
  const confidence = (predictions[maxIndex] * 100).toFixed(2);

  return `Classe ${maxIndex} - Confiança: ${confidence}%`;
}




async function main() {
  await setupWebcam();
  await loadModel();
  predictLoop();
}

main();
