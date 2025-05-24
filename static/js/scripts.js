function closeNotification(notificationId) {
    document.getElementById(notificationId).style.display = "none";
}

function initMap() {
    // Create a map object
    var map = new google.maps.Map(document.getElementById('map'), {
      center: { lat: 6.9271, lng: 79.8612 }, // Coordinates for your location
      zoom: 10
    });

    // Add a marker on the map
    var marker = new google.maps.Marker({
      position: { lat: 6.9271, lng: 79.8612 },
      map: map,
      title: 'My Location',
    });
}