import imageio


def create_gif(image_list, gif_name):  # checked
    r'''

    Args:
        image_list: a list of image path
        gif_name: the name of gif you want to generate

    Returns:
    Examples:
        image_list = []
        for i in range(13, 26):
            image_list.append('graph_save/undirecetd_weighted_network_%d.jpg' % i)

        gif_name = 'undirecetd_weighted_network.gif'
        create_gif(image_list, gif_name)
    '''
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=1)  # stop time
    return


if __name__ == "__main__":
    image_list = []
    for i in range(1, 10):
        image_list.append('save_fig/cifar10_cResNet18_undirecetd_weighted_network_%d.jpg' % i)

    gif_name = 'cifar10_cResNet18_undirecetd_weighted_network.gif'
    create_gif(image_list, gif_name)