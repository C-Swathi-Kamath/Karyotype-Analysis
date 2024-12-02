$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();
    $('#xgResult').hide();
    $('#rfResult').hide();
    $('#nbResult').hide();
    $('#svmResult').hide();

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
            reader.readAsDataURL(input.files[1]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#class_info').text('');

        $('#result').hide();
        $('#xgResult').hide();
        $('#rfResult').hide();
        $('#nbResult').hide();
        $('#svmResult').hide();
        readURL(this);
    });

    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {

                // Get and display the result
                result = data.result
                xgResult = data.xgResults
                rfResult = data.rFResults
                nbResult = data.nBResults
                svmResult = data.svmResults
                $('.loader').hide();
                $('#result').fadeIn(600);
                $('#xgResult').fadeIn(600);
                $('#svmResult').fadeIn(600);
                $('#rfResult').fadeIn(600);
                $('#nbResult').fadeIn(600);
                $('#class_info').fadeIn(600);
                $('#result').text("Inception Prediction : "+ result);
                $('#xgResult').text("XgBoost Prediction : "+ xgResult);
                $('#svmResult').text("SVM Prediction : "+ svmResult);
                $('#rfResult').text("Random Forest Prediction : "+ rfResult);
                $('#nbResult').text("Naive Bayes Prediction : "+ nbResult);
                console.log('Success!');
            },
        });
    });

});
