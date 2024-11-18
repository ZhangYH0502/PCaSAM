import torch


def dfs(binary_image, visited, x, y, component):
    # 检查当前像素是否在图像范围内且为1，并且没有被访问过
    if x < 0 or y < 0 or x >= binary_image.size(0) or y >= binary_image.size(1) \
            or binary_image[x, y] == 0 or visited[x, y]:
        return
    # 将当前像素标记为已访问
    visited[x, y] = True
    # 将当前像素加入到当前连通域中
    component.append((x, y))
    # 深度优先搜索四个方向的相邻像素
    dfs(binary_image, visited, x + 1, y, component)
    dfs(binary_image, visited, x - 1, y, component)
    dfs(binary_image, visited, x, y + 1, component)
    dfs(binary_image, visited, x, y - 1, component)


def find_connected_components(binary_image):
    visited = torch.zeros_like(binary_image, dtype=torch.bool)
    components = []

    for i in range(binary_image.size(0)):
        for j in range(binary_image.size(1)):
            if binary_image[i, j] == 1 and not visited[i, j]:
                component = []
                # 对当前连通域进行深度优先搜索
                dfs(binary_image, visited, i, j, component)
                components.append(component)

    # 计算每个连通域的左上角和右下角坐标
    component_coords = [((min(x for x, _ in component), min(y for _, y in component)),
                         (max(x for x, _ in component), max(y for _, y in component)))
                        for component in components]

    return len(components), component_coords


if __name__ == '__main__':
    # 示例输入的二值化图像
    binary_image = torch.tensor([[0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0]])

    num_components, component_coords = find_connected_components(binary_image)

    print("Number of connected components:", num_components)
    print("Coordinates of each connected component:")
    for idx, ((x1, y1), (x2, y2)) in enumerate(component_coords):
        print(f"Component {idx + 1}: Top left ({x1}, {y1}), Bottom right ({x2}, {y2})")
