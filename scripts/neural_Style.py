import functools
import time
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import IPython.display
mpl.rcParams['figure.figsize'] = (10, 10)
mpl.rcParams['axes.grid'] = False


class StyleTransfer:
    """
    """

    # Content layer where will pull our feature maps
    content_layers = ['block5_conv2']

    # Style layer we are interested in
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1'
                    ]

    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)

    def load_img(self, path_to_img):
        max_dim = 512
        img = Image.open(path_to_img)
        long = max(img.size)
        scale = max_dim/long
        img = img.resize(
            (round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS)

        img = kp_image.img_to_array(img)

        # We need to broadcast the image array such that it has a batch dimension
        img = np.expand_dims(img, axis=0)
        return img

    def imshow(self, img, title=None):
        # Remove the batch dimension
        out = np.squeeze(img, axis=0)
        # Normalize for display
        out = out.astype('uint8')
        plt.imshow(out)
        if title is not None:
            plt.title(title)
        plt.imshow(out)

    def load_and_process_img(self, path_to_img):
        img = load_img(path_to_img)
        img = tf.keras.applications.vgg19.preprocess_input(img)
        return img

    def deprocess_img(self, processed_img):
        x = processed_img.copy()
        if len(x.shape) == 4:
            x = np.squeeze(x, 0)
        assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                                   "dimension [1, height, width, channel] or [height, width, channel]")
        if len(x.shape) != 3:
            raise ValueError("Invalid input to deprocessing image")

        # perform the inverse of the preprocessiing step
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68
        x = x[:, :, ::-1]

        x = np.clip(x, 0, 255).astype('uint8')
        return x

    def gram_matrix(self, input_tensor):
        # We make the image channels first
        channels = int(input_tensor.shape[-1])
        a = tf.reshape(input_tensor, [-1, channels])
        n = tf.shape(a)[0]
        gram = tf.matmul(a, a, transpose_a=True)
        return gram / tf.cast(n, tf.float32)

    def get_style_loss(self, base_style, gram_target):
        """Expects two images of dimension h, w, c"""
        # height, width, num filters of each layer
        # We scale the loss at a given layer by the size of the feature map and the number of filters
        height, width, channels = base_style.get_shape().as_list()
        gram_style = gram_matrix(base_style)
        
        return tf.reduce_mean(tf.square(gram_style - gram_target))# / (4. * (channels ** 2) * (width * height) ** 2)


    def get_feature_representations(self, model, content_path, style_path):
        """Helper function to compute our content and style feature representations.

        This function will simply load and preprocess both the content and style 
        images from their path. Then it will feed them through the network to obtain
        the outputs of the intermediate layers. 
        
        Arguments:
            model: The model that we are using.
            content_path: The path to the content image.
            style_path: The path to the style image
            
        Returns:
            returns the style features and the content features. 
        """
        # Load our images in
        content_image = load_and_process_img(content_path)
        style_image = load_and_process_img(style_path)

        # batch compute content and style features
        style_outputs = model(style_image)
        content_outputs = model(content_image)

        # Get the style and content feature representations from our model
        style_features = [style_layer[0]
                            for style_layer in style_outputs[:num_style_layers]]
        content_features = [content_layer[0]
                            for content_layer in content_outputs[num_style_layers:]]
        return style_features, content_features


    def run_style_transfer(self, content_path,
                        style_path,
                        num_iterations=1000,
                        content_weight=1e3,
                        style_weight=1e-2):
        # We don't need to (or want to) train any layers of our model, so we set their
        # trainable to false.
        model = get_model()
        for layer in model.layers:
            layer.trainable = False

        # Get the style and content feature representations (from our specified intermediate layers)
        style_features, content_features = get_feature_representations(
            model, content_path, style_path)
        gram_style_features = [gram_matrix(style_feature)
                                for style_feature in style_features]

        # Set initial image
        init_image = load_and_process_img(content_path)
        init_image = tf.Variable(init_image, dtype=tf.float32)
        # Create our optimizer
        opt = tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)

        # For displaying intermediate images
        iter_count = 1

        # Store our best result
        best_loss, best_img = float('inf'), None

        # Create a nice config
        loss_weights = (style_weight, content_weight)
        cfg = {
            'model': model,
            'loss_weights': loss_weights,
            'init_image': init_image,
            'gram_style_features': gram_style_features,
            'content_features': content_features
        }

        # For displaying
        num_rows = 2
        num_cols = 5
        display_interval = num_iterations/(num_rows*num_cols)
        start_time = time.time()
        global_start = time.time()

        norm_means = np.array([103.939, 116.779, 123.68])
        min_vals = -norm_means
        max_vals = 255 - norm_means

        imgs = []
        for i in range(num_iterations):
            grads, all_loss = compute_grads(cfg)
            loss, style_score, content_score = all_loss
            opt.apply_gradients([(grads, init_image)])
            clipped = tf.clip_by_value(init_image, min_vals, max_vals)
            init_image.assign(clipped)
            end_time = time.time()

            if loss < best_loss:
                # Update best loss and best image from total loss.
                best_loss = loss
                best_img = deprocess_img(init_image.numpy())

            if i % display_interval == 0:
                tart_time = time.time()

                # Use the .numpy() method to get the concrete numpy array
                plot_img = init_image.numpy()
                plot_img = deprocess_img(plot_img)
                imgs.append(plot_img)
                IPython.display.clear_output(wait=True)
                IPython.display.display_png(Image.fromarray(plot_img))
                print('Iteration: {}'.format(i))
                print('Total loss: {:.4e}, '
                        'style loss: {:.4e}, '
                        'content loss: {:.4e}, '
                        'time: {:.4f}s'.format(loss, style_score, content_score, time.time() - start_time))
        print('Total time: {:.4f}s'.format(time.time() - global_start))
        IPython.display.clear_output(wait=True)
        plt.figure(figsize=(14, 4))
        for i, img in enumerate(imgs):
            plt.subplot(num_rows, num_cols, i+1)
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])

        return best_img, best_loss


    def show_results_a(self, best_img, content_path,
                    style_path,
                    output_path,
                    combined_path,
                    width, height,
                    content_dump_path=INPUT_IMAGE_PATH,
                    style_dump_path=STYLE_IMAGE_PATH,
                    output_dump_path=OUTPUT_PATH,
                    show_large_final=True):
        f = plt.figure(figsize=(10, 15))
        content = load_img(content_path)
        style = load_img(style_path)

        plt.subplot(1, 2, 1)
        imshow(content, 'Content Image')

        plt.subplot(1, 2, 2)
        imshow(style, 'Style Image')

        if show_large_final:
            plt.figure(figsize=(10, 10))
            #plt.subplot(3,3,3)
            plt.imshow(best_img)
            plt.title('Output Image')

            plt.show()
        refactor(content_path, style_path, output_path, content_dump_path,
                style_dump_path, output_dump_path, width, height)

        IMAGE_SIZE = 500
        IMAGE_WIDTH = width
        IMAGE_HEIGHT = height

        combined_marble = Image.new("RGB", (IMAGE_WIDTH*3, IMAGE_HEIGHT))
        x_offset = 0
        for image in map(Image.open, [content_dump_path, style_dump_path, output_dump_path]):
            combined_marble.paste(image, (x_offset, 0))
            x_offset += IMAGE_WIDTH
        combined_marble.save(combined_path)
        print("path of combined image: ", combined_path)
        return combined_marble

    def refactor(self,content_path, style_path, output_path, content_dump_path,
             style_dump_path, output_dump_path, width, height):
        '''
        content_path: content path
        style_path: style path
        output_path: styled image after optimization
        *_dump_path: bins for images
        width: w of image
        height: h of image 

        '''

        collections = [(content_path, content_dump_path), (style_path, style_dump_path),
                        (output_path, output_dump_path)]
        for image in collections:
            print(image)
            process_image = Image.open(image[0])
            process_image = process_image.resize((width, height))
            process_image.save(image[1])
            process_image

