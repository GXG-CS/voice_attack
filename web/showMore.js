function keepClickingShowMore() {
    var showMoreButton = document.querySelector('.full-width-message.clickable');
    
    if (showMoreButton) {
        console.log('Show more button found, attempting to click.');
        showMoreButton.click();
        console.log('Clicked "Show more".');
    } else {
        console.log('No more "Show more" button found, or waiting for content to load.');
        // Don't clear the interval here, in case content is still loading
    }
}

// Increase the interval if needed to allow content to load
var clickInterval = setInterval(keepClickingShowMore, 3000); // Click every 3 seconds

// To stop the interval use:
// clearInterval(clickInterval);