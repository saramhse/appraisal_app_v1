// Creating map object
var myMap = L.map("map", {
  center: [33.6582614,  -117.8230375],
  zoom: 11
});

// Adding tile layer to the map
L.tileLayer("https://api.tiles.mapbox.com/v4/{id}/{z}/{x}/{y}.png?access_token={accessToken}", {
  attribution: "Map data &copy; <a href=\"https://www.openstreetmap.org/\">OpenStreetMap</a> contributors, <a href=\"https://creativecommons.org/licenses/by-sa/2.0/\">CC-BY-SA</a>, Imagery © <a href=\"https://www.mapbox.com/\">Mapbox</a>",
  maxZoom: 18,
  id: "mapbox.streets",
  accessToken: API_KEY
}).addTo(myMap);



var markers =  L.markerClusterGroup();
geodata.features.map((data) => {
    //console.log(data)
    var location = data.geometry;
    if(location){
        if(location.coordinates.length > 0){
            console.log(location)
            markers.addLayer(L.marker([location.coordinates[1], location.coordinates[0]])
            .bindPopup("Address: " + data.properties.full_address + "<br>Beds:" + data.properties.beds + "  " + "Baths:" + data.properties.baths + "<br>Price:$" + data.properties.price + "<br>HOA:$" + data.properties.hoa + "<br>Year Built:" + data.properties.built + "<br>Sq feet:" + data.properties.sqrft))
        }
    }
})

myMap.addLayer(markers);
