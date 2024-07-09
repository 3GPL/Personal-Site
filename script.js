document.addEventListener("DOMContentLoaded", function() {
    // Define the neural network architecture
    const inputLayerSize = 3;
    const hiddenLayerSize = 3;
    const outputLayerSize = 1;

    // Initialize weights with random values
    let weights1 = Array.from({ length: inputLayerSize }, () =>
        Array.from({ length: hiddenLayerSize }, () => Math.random())
    );
    let weights2 = Array.from({ length: hiddenLayerSize }, () =>
        Array.from({ length: outputLayerSize }, () => Math.random())
    );

    // Activation function (ReLU)
    function relu(x) {
        return x.map(value => Math.max(0, value));
    }

    // Derivative of ReLU
    function reluDerivative(x) {
        return x.map(value => (value > 0 ? 1 : 0));
    }

    // Forward propagation
    function forwardPropagation(input) {
        const hiddenInput = input.map((inp, i) => 
            weights1[i].reduce((sum, weight, j) => sum + inp * weight, 0)
        );
        const hiddenOutput = relu(hiddenInput);
        
        const finalInput = hiddenOutput.map((hiddenOut, i) => 
            weights2[i].reduce((sum, weight, j) => sum + hiddenOut * weight, 0)
        );
        const output = relu(finalInput);

        return { hiddenOutput, output };
    }

    // Backward propagation (simplified)
    function backwardPropagation(input, hiddenOutput, output, expectedOutput) {
        const outputError = output.map((out, i) => expectedOutput[i] - out);
        const hiddenError = hiddenOutput.map((hiddenOut, i) => 
            reluDerivative(hiddenOut) * weights2[i].reduce((sum, weight, j) => sum + outputError[j] * weight, 0)
        );

        // Update weights (using a learning rate)
        const learningRate = 0.01;
        for (let i = 0; i < hiddenLayerSize; i++) {
            for (let j = 0; j < outputLayerSize; j++) {
                weights2[i][j] += learningRate * hiddenOutput[i] * outputError[j];
            }
        }
        for (let i = 0; i < inputLayerSize; i++) {
            for (let j = 0; j < hiddenLayerSize; j++) {
                weights1[i][j] += learningRate * input[i] * hiddenError[j];
            }
        }
    }

    // Simple training loop
    function train(input, expectedOutput, epochs) {
        for (let epoch = 0; epoch < epochs; epoch++) {
            const { hiddenOutput, output } = forwardPropagation(input);
            backwardPropagation(input, hiddenOutput, output, expectedOutput);
        }
    }

    // Event listener for the Train button
    document.getElementById('trainButton').addEventListener('click', function() {
        const input = [
            parseFloat(document.getElementById('inputValue1').value),
            parseFloat(document.getElementById('inputValue2').value),
            parseFloat(document.getElementById('inputValue3').value)
        ];
        const expectedOutput = [1]; // You can change this to the desired output

        // Train the network
        train(input, expectedOutput, 1000);

        alert("Training complete");
    });

    // Event listener for the Predict button
    document.getElementById('predictButton').addEventListener('click', function() {
        const input = [
            parseFloat(document.getElementById('inputValue1').value),
            parseFloat(document.getElementById('inputValue2').value),
            parseFloat(document.getElementById('inputValue3').value)
        ];

        // Predict the output
        const result = forwardPropagation(input).output;
        document.getElementById('result').innerText = `Output: ${result}`;
    });
});
