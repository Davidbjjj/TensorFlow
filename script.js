const URL = "model/"; // Caminho onde está seu model.json e metadata.json
let model, webcam, labelContainer, maxPredictions;

async function init() {
  const modelURL = URL + "model.json";
  const metadataURL = URL + "metadata.json";

  // Carrega o modelo e o metadata
  model = await tmImage.load(modelURL, metadataURL);
  maxPredictions = model.getTotalClasses();

  // Configura webcam
  const flip = true; // espelha a imagem horizontalmente
  webcam = new tmImage.Webcam(200, 200, flip);
  await webcam.setup(); // pede permissão da câmera
  await webcam.play();
  window.requestAnimationFrame(loop);

  document.getElementById("webcam-container").appendChild(webcam.canvas);

  labelContainer = document.getElementById("label-container");
  for (let i = 0; i < maxPredictions; i++) {
    labelContainer.appendChild(document.createElement("div"));
  }
}

async function loop() {
  webcam.update(); // atualiza a imagem da webcam
  await predict();
  window.requestAnimationFrame(loop);
}

async function predict() {
  const prediction = await model.predict(webcam.canvas);
  for (let i = 0; i < maxPredictions; i++) {
    const classPrediction =
      prediction[i].className + ": " + (prediction[i].probability * 100).toFixed(2) + "%";
    labelContainer.childNodes[i].innerText = classPrediction;
  }
}

// Inicia o app
init();
