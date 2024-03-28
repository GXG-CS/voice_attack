// Step 1: Extract the text
var contentBoxes = Array.from(document.querySelectorAll('.apd-content-box.with-activity-page'));
var allText = contentBoxes.map(box => box.innerText).join("\n\n");

// Step 2: Trigger the download
var blob = new Blob([allText], { type: 'text/plain' });
var link = document.createElement('a');
link.href = window.URL.createObjectURL(blob);
link.download = 'alexa_activity.txt';
link.click();