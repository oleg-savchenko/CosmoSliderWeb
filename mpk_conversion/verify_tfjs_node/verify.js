// Verification layer 3: load the ACTUAL shipped web_model/mpk_lin and mpk_nonlin
// artifacts via tf.loadGraphModel (the exact loader/runtime the website itself uses)
// and compare against the real CosmoPowerJAX-generated test vectors.
//
// Run: cd verify_tfjs_node && npm install && node verify.js

const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');

const TOL = 1e-3;

function maxRelErr(pred, truth) {
  let maxErr = 0;
  for (let i = 0; i < pred.length; i++) {
    const err = Math.abs(pred[i] - truth[i]) / Math.abs(truth[i]);
    if (err > maxErr) maxErr = err;
  }
  return maxErr;
}

async function verifyModel(name, modelPath, xData, yData) {
  const model = await tf.loadGraphModel(`file://${modelPath}`);
  const input = tf.tensor(xData);
  const output = model.predict(input);
  const predArray = await output.array();

  let worst = 0;
  for (let i = 0; i < predArray.length; i++) {
    const err = maxRelErr(predArray[i], yData[i]);
    if (err > worst) worst = err;
  }
  console.log(`${name}: max relative error vs JAX (via tf.loadGraphModel) = ${worst.toExponential(3)}`);

  input.dispose();
  output.dispose();

  if (worst >= TOL) {
    throw new Error(`${name}: error ${worst} exceeds tolerance ${TOL}`);
  }
  return worst;
}

async function main() {
  const vectors = JSON.parse(
    fs.readFileSync(path.join(__dirname, '..', 'verification_data', 'vectors.json'), 'utf8')
  );

  await verifyModel(
    'mpk_lin',
    path.join(__dirname, '..', '..', 'web_model', 'mpk_lin', 'model.json'),
    vectors.x_lin,
    vectors.y_lin
  );

  await verifyModel(
    'mpk_nonlin',
    path.join(__dirname, '..', '..', 'web_model', 'mpk_nonlin', 'model.json'),
    vectors.x_nonlin,
    vectors.y_nonlin
  );

  console.log(`\nLayer 3 PASSED (all errors < ${TOL})`);
}

main().catch((err) => {
  console.error('Layer 3 FAILED:', err.message);
  process.exit(1);
});
