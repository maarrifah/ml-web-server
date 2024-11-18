const tfjs = require('@tensorflow/tfjs-node');

// load model tensorflow.js
function loadModel() {
    const modelUrl = "file://models/model.json";
    return tfjs.loadLayersModel(modelUrl);
  }

// predict() untuk memprediksi data (berupa imageBuffer) dengan model yang telah di-load
function predict(model, imageBuffer) {
    const tensor = tfjs.node
      .decodeJpeg(imageBuffer)
      .resizeNearestNeighbor([150, 150])
      .expandDims()
      .toFloat();
   
    return model.predict(tensor).data();
  }

  module.exports = { loadModel, predict };