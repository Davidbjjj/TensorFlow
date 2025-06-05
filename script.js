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
  const tensor = tf.browser.fromPixels(webcamElement)
    .resizeNearestNeighbor([224, 224])
    .toFloat()
    .expandDims();

  const predictions = await model.predict(tensor).data();

  // Pegando o índice da maior previsão
  const maxIndex = predictions.indexOf(Math.max(...predictions));
  const confidence = (predictions[maxIndex] * 100).toFixed(2);

  return `Classe ${maxIndex + 1} - Confiança: ${confidence}%`;
}

async function main() {
  await setupWebcam();
  await loadModel();
  predictLoop();
}

main();
