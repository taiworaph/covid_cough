/**
 * Uploading a waveform cough file for  covid-19 detection
 */
function handleUpload() {

    $('#btnUpload').on('click', function (e) {
        $("#dvMessage").text("").hide();
        $("#dvLoader").prepend($('<img>', {src: '/static/img/loading.gif'}));
        let startTime = Date.now();
        e.preventDefault();
        $(this).prop('disabled', true);

        $("#spnStatus").removeClass("invisible").show();
        $('#frmUpload').submit();
        $('#flUpload').prop('disabled', true);
    });
}

function showPageSpinner(){
    $('body').append($('<div />', { "class": "loading"}));
}

