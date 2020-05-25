# Mnist-Model-Deploy

[Original file is located at Google Colab](https://colab.research.google.com/github/Vatsalparsaniya/Mnist-Model-Deploy/blob/master/__init__.ipynb)

## WebApp using Flask with Google Colab of MNIST model Deploy 

#### Handle Post request using Flask
    @app.route('/mnistprediction/', methods=['GET', 'POST'])
    def mnist_prediction():
        if request.method == "POST":
            if not request.files['file'].filename:
                flash("No File Found")
            else:
                f =  request.files['file']
                f.save("uploads/"+f.filename)
                image_gray  = cv2.imread("uploads/"+f.filename, cv2.IMREAD_GRAYSCALE)
                img_resize = cv2.resize(image_gray,(28,28))
                image_bw = cv2.threshold(img_resize, 75, 255, cv2.THRESH_BINARY)[1]
                bitwise_not_image = cv2.bitwise_not(image_bw, mask=None)
                pred_img = np.reshape(bitwise_not_image,(1,28,28,1))/255.0
                predictions = mnist_model.predict(pred_img)
                number = int(np.argmax(predictions))
                plt.figure()
                y_pos = np.arange(10)
                plt.bar(y_pos, predictions[0])
                plt.savefig('generated_image/mnist_vis/'+f.filename)
                return str(number)
                
#### Prediction Button AJAX request

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
                      $('#resultModel').text(' Predicted Number :  ' + data);
                      console.log(fileName)
                      $('#image_div1').attr('src','/get-mnist-image/'+fileName)
                  },
              });
          });    
      });
      
 
![MNIST Webapp](https://github.com/Vatsalparsaniya/Mnist-Model-Deploy/blob/master/static/image/mnist.PNG)
