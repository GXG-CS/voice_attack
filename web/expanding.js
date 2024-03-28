// Select all the buttons responsible for expanding the details
var expandButtons = document.querySelectorAll('.apd-expand-toggle-button');

// Click each button to expand the details if they are not already expanded
expandButtons.forEach(function(button) {
    // Check if the button has a class that indicates collapsed state, for example 'fa-chevron-down'
    if (button.classList.contains('fa-chevron-down')) {
        button.click();
    }
});