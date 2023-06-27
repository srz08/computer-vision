import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def line_detection_non_vectorized(edge_image, num_rhos=180, num_thetas=180, t_count=60, filter=True):
    image_height = edge_image.shape[0]
    image_width = edge_image.shape[1]
    upper_rho_limit = np.sqrt(np.square(image_height) + np.square(image_width))
    lower_rho_limit = -upper_rho_limit

    theta_grid = np.linspace(0.0, np.pi, num_thetas)
    cos_theta = np.cos(theta_grid)
    sin_theta = np.sin(theta_grid)

    rho_grid = np.linspace(lower_rho_limit, upper_rho_limit, num_rhos)

    voting_table = np.zeros((num_rhos, num_thetas), dtype=np.uint64)

    plot_x = []
    plot_y = []
    for y in range(image_height):
        for x in range(image_width):
            if edge_image[y][x] != 0:
                all_y = []
                all_x = []
                for i in range(len(theta_grid)):
                    rho = (x * cos_theta[i]) + (y * sin_theta[i])
                    voting_table[np.argmin(np.abs(rho_grid - rho))][i] += 1
                    all_y.append(rho)
                    all_x.append(theta_grid[i])
                plot_x.append(all_x)  
                plot_y.append(all_y)

    lists = []
    all_theta = []
    all_rho = []
    to_filter = {}
    for y in range(voting_table.shape[0]):
      for x in range(voting_table.shape[1]):
        if voting_table[y][x] > t_count:
          if not filter:
            rho, theta  = rho_grid[y], theta_grid[x]
            cos, sin = cos_theta[x], sin_theta[x]
            all_theta.append(theta)
            all_rho.append(rho)
            lists.append([int((cos * rho)  + 1000 * (-sin)),int((sin * rho)  + 1000 * (cos)),int((cos * rho)  - 1000 * (-sin)),int((sin * rho)  - 1000 * (cos))])
          to_filter[(x, y)] = voting_table[y][x]

    if filter:
      theta_limit = num_thetas/20
      rho_limit = num_rhos/20
      for (x1, y1) in to_filter:
        for (x2, y2) in to_filter:
          if (x1, y1) != (x2, y2):
            if abs(x1 - x2) < theta_limit and abs(y1 - y2) < rho_limit:
              if to_filter[(x1, y1)] > to_filter[(x2, y2)]:
                to_filter[(x2, y2)] = 0
              else:
                to_filter[(x1, y1)] = 0

      for (x, y) in to_filter:
        if to_filter[(x, y)] != 0:
          rho, theta  = rho_grid[y], theta_grid[x]
          cos, sin = cos_theta[x], sin_theta[x]
          all_theta.append(theta)
          all_rho.append(rho)
          lists.append([int((cos * rho)  + 1000 * (-sin)),int((sin * rho)  + 1000 * (cos)),int((cos * rho)  - 1000 * (-sin)),int((sin * rho)  - 1000 * (cos))])

    
    return lists, all_theta, all_rho, plot_x, plot_y




if __name__ == "__main__":
    image = cv2.imread("test.bmp")
    edge_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge_image = cv2.Canny(edge_image, 100, 200)
    num_rhos = 180
    num_thetas = 170
    t_count = 70
    lists, all_theta, all_rho, plot_x, plot_y = line_detection_non_vectorized(edge_image, num_rhos, num_thetas, t_count, filter=False)
    black_image = np.zeros_like(image)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize = (16,4))
    #add titles to the figure
    fig.suptitle("Hough Transform with {} rhos, {} thetas, and threshold {}".format(num_rhos, num_thetas, t_count))
    ax1.imshow(image)
    ax1.title.set_text("Image")
    ax2.imshow(edge_image, cmap="gray")
    ax2.title.set_text("Edges")
    ax3.title.set_text("Representation in Space")
    ax3.set_facecolor((0, 0, 0))
    ax4.title.set_text("Hough Transform Output")
    ax4.imshow(image)

    for all_x, all_y in zip(plot_x, plot_y):
        ax3.plot(all_x, all_y)
    for x1, y1, x2, y2 in lists:
        ax4.add_line(mlines.Line2D([x1, x2], [y1, y2]))
    for t, r in zip(all_theta, all_rho):
        plot1 = [t]
        plot2 = [r] 
        ax3.plot(plot1, plot2)      
    
    plt.show()

