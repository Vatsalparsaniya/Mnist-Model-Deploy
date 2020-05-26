// var fileName = 50

// $(document).ready(function(){
//     $('input[type="file"]').change(function(e){
//         fileName = e.target.files[0].name;
//         // alert('The file "' + fileName +  '" has been selected.');
//     });
// });


$(document).ready(function(){
    $('#predbtn').click(function () {
        var form_data = new FormData($('#upload-file-model')[0]);
        
        $.ajax({
            type: 'POST',   
            url: '/mnistprediction/',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data){
                // Get and display the result
                // $('#resultModel').fadeIn(600);

                $('#resultModel').text(' Predicted Number :  ' + data);
                $('#image_div1').attr('src','/get-mnist-image/generated_bar.PNG');
            },
        });
    });    
});