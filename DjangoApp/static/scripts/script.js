function btn_click() {
    var model = document.getElementById("models").value;
    var instrument = document.getElementById("instruments").value;
    location.href=`/song/${model}/${instrument}`
}
function btn_click2() {
    var model = document.getElementById("models2").value;
    var instrument = document.getElementById("instruments2").value;
    location.href=`/generate/${model}/${instrument}`
}