document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('prediction-form');
    const resultDiv = document.getElementById('result');
    const predictionText = document.getElementById('prediction-text');
    const probabilityText = document.getElementById('probability-text');

    form.addEventListener('submit', function(e) {
        e.preventDefault();

        const formData = new FormData(form);
        const data = Object.fromEntries(formData);

        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        })
        .then(response => response.json())
        .then(result => {
            resultDiv.classList.remove('hidden');
            predictionText.textContent = result.prediction === 1 ? 'Diabetes detected' : 'No diabetes detected';
            probabilityText.textContent = `Probability: ${(result.probability * 100).toFixed(2)}%`;
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while making the prediction. Please try again.');
        });
    });
});