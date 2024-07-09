document.addEventListener("DOMContentLoaded", function() {
    // Define a sequential model
    const model = tf.sequential();

    // Add an input layer
    model.add(tf.layers.dense({units: 3, inputShape: [3], activation: 'relu'}));

    // Add a hidden layer
    model.add(tf.layers.dense({units: 3, activation: 'relu'}));

    // Add an output layer
    model.add(tf.layers.dense({units: 1, activation: 'sigmoid'}));

    // Compile the model
    model.compile({optimizer: 'adam', loss: 'binaryCrossentropy'});

    console.log("Neural network created and compiled successfully.");
});
