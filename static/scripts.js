document.addEventListener("DOMContentLoaded", function() {
    const form = document.querySelector("form");
    form.addEventListener("submit", function(event) {
        event.preventDefault(); // Prevent the form from submitting via the browser
        const input = document.querySelector('input[name="question"]').value;
        fetch('/general', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `question=${encodeURIComponent(input)}`
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById('response').innerHTML = `<p>${data.response}</p>`;
        })
        .catch(error => console.error('Error:', error));
    });
});
