__author__ = 'enlightenedcsf'

import math
import numpy as np
import cv2


def nothing(x):
    pass


def compare(rho1, theta1, rho2, theta2):
    # return False
    return (abs(rho1 - rho2) <= 10) and (abs(theta1 - theta2) < 0.3)


def find_unique_lines(input_lines):
    if (input_lines is None) or (len(input_lines) == 0):
        return None
    if len(input_lines[0]) == 1:
        return input_lines
    new_lines = [input_lines[0][0]]

    for r_in, t_in in input_lines[0]:
        # if r_in < 0:
        #     continue
        to_add = True
        for rn, tn in new_lines:
            if compare(r_in, t_in, rn, tn):
                to_add = False
                break
        if to_add:
            new_lines.append([r_in, t_in])

    # new_abc_lines = []
    # for r, t in new_lines:
    #     new_abc_lines.append(get_a_b_c(r, t))

    return new_lines


def get_a_b_c(line_rho, line_theta):
    _a = np.cos(line_theta)
    _b = np.sin(line_theta)
    _x0 = _a * line_rho
    _y0 = _b * line_rho
    _x1 = int(_x0 + 1000 * (-_b))
    _y1 = int(_y0 + 1000 * _a)
    _x2 = int(_x0 - 1000 * (-_b))
    _y2 = int(_y0 - 1000 * _a)

    _delta_y = _y2 - _y1
    _delta_x = _x2 - _x1

    if _delta_x == 0:
        return 1, 0, -line_rho  # x = something, y is zero
    if _delta_y == 0:
        return 0, -1, line_rho  # y = something, x is zero

    _out_a = _y2 - _y1
    _out_b = _x1 - _x2
    _out_c = _y1 * _x2 - _x1 * _y2
    return _out_a, _out_b, _out_c


def transform_lines(rho_theta_lines):
    abc_lines = []
    for line in rho_theta_lines:
        abc_lines.append(get_a_b_c(line[0], line[1]))
    return abc_lines


def find_intersections(new_lines):
    points = []
    for new_rho, new_theta in new_lines:
        line1abc = get_a_b_c(new_rho, new_theta)
        for new_rho2, new_theta2 in new_lines:
            if (new_rho == new_rho2) and (new_theta == new_theta2):
                continue
            line2abc = get_a_b_c(new_rho2, new_theta2)
            Mx = [[-line1abc[2], line1abc[1]], [-line2abc[2], line2abc[1]]]
            My = [[line1abc[0], -line1abc[2]], [line2abc[0], -line2abc[2]]]
            detM = np.linalg.det([[line1abc[0], line1abc[1]], [line2abc[0], line2abc[1]]])

            if detM == 0:
                continue

            x = np.linalg.det(Mx) / detM
            y = np.linalg.det(My) / detM
            if not (np.isnan(x) or np.isnan(y) or np.isinf(x) or np.isinf(y)):
                points.append((x, y))
    return points


def are_close(point1, point2):
    return math.sqrt(
        (point2[0] - point1[0]) * (point2[0] - point1[0]) + (point2[1] - point1[1]) * (point2[1] - point1[1])) < 15


def find_unique_points(input_points):
    final_points = [input_points[0]]
    for in_point in input_points[1:]:
        to_add = True
        for fin_point in final_points:
            if are_close(in_point, fin_point):
                to_add = False
                break
        if to_add:
            final_points.append(in_point)
    return final_points


def sort_points(points):
    for i in range(1, len(points)-1):
        current = points[i]
        prev_key = i-1
        while prev_key >= 0 and (points[prev_key][1] > current[1] or        # [1] - comparison by y-component
                                (points[prev_key][1] == current[1] and points[prev_key][0] > current[0])): # if y's are equal, then compare x's
            points[prev_key+1] = points[prev_key]
            points[prev_key] = current
            prev_key -= 1


def get_points_on_line(line, input_points):
    new_points = []
    for p in input_points:
        value_in_point = line[0]*p[0] + line[1]*p[1] + line[2]
        if abs(value_in_point) < 1:
            new_points.append(p)
    sort_points(new_points)
    return new_points


def are_points_connected(point1, point2, line):
    test_point = ((point1[0] + point2[0])/2, (point1[1] + point2[1])/2)
    return is_point_dark(test_point)


def is_point_dark(point):
    if point[1] < 0 or point[0] < 0:
        return False
    density = []
    if point[0] <= img.shape[1] and point[1] <= img.shape[0]:
        density.append(thres[point[1],   point[0]])
        # print("{:.1f}".format(point[0]), "{:.1f}".format(point[1]), density[0])
        # if density < 160:
        if density[0] < 255:
            return True

        density.append(thres[point[1]+3, point[0]])
        density.append(thres[point[1]-3, point[0]])
        density.append(thres[point[1],   point[0]+3])
        density.append(thres[point[1],   point[0]-3])
        density.append(thres[point[1]+2, point[0]+2])
        density.append(thres[point[1]+2, point[0]-2])
        density.append(thres[point[1]-2, point[0]+2])
        density.append(thres[point[1]-2, point[0]-2])

        density.append(thres[point[1]-3, point[0]-3])
        density.append(thres[point[1]-3, point[0]+3])
        density.append(thres[point[1]+3, point[0]-3])
        density.append(thres[point[1]+3, point[0]+3])

        density.append(thres[point[1]-4, point[0]])
        density.append(thres[point[1]+4, point[0]])
        density.append(thres[point[1], point[0]-4])
        density.append(thres[point[1], point[0]+4])

        cnt = 0
        for d in density:
            if d < 255:
                cnt += 1
        return cnt > 1

    else:
        return False


def are_points_close(p1, p2):
    return math.sqrt((p2[0] - p1[0])*(p2[0] - p1[0]) + (p2[1] - p1[1])*(p2[1] - p1[1])) < 0.1


def extract_vertices_and_edges(input_points, input_lines):
    edges = []
    for line in input_lines:
        start_new_edge = True
        points_on_line = get_points_on_line(line, input_points)
        points_on_edge = []
        for p in points_on_line:
            if is_point_dark(p):
                points_on_edge.append(p)
        if len(points_on_edge) == 1:
            continue
        for i in range(0, len(points_on_edge)-2):
            if are_points_connected(points_on_edge[i], points_on_edge[i+1], line):
                if are_points_close(points_on_edge[i], points_on_edge[i+1]):
                    continue
                # if len(edges) == 0:
                #     edges.append((points_on_edge[i], points_on_edge[i+1]))
                #     start_new_edge = False
                if start_new_edge:
                    edges.append([points_on_edge[i], points_on_edge[i+1]])
                    start_new_edge = False
                # if last point on a current edge is equal to a start one on the interval
                elif edges[len(edges)-1][1] == points_on_edge[i]:
                    edges[len(edges)-1][1] = points_on_edge[i+1]
                else:
                    edges.append([points_on_edge[i], points_on_edge[i+1]])
    return edges


########################################################################################################################
cv2.namedWindow('image')
# cv2.namedWindow('gray')

cv2.createTrackbar('rho', 'image', 1, 20, nothing)
cv2.createTrackbar('theta', 'image', 30, 180, nothing)
cv2.createTrackbar('threshold', 'image', 50, 250, nothing)
cv2.setTrackbarPos('threshold', 'image', 250)

img = cv2.imread("/home/enlightenedcsf/opencv/pyTest2/test2.jpg")
# print img.shape
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = cv2.GaussianBlur(gray, (5, 5), 0)
thres = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 39, 25)

edges = cv2.Canny(thres, 100, 200)

while 1:
    cv2.imshow('image', img)
    # cv2.imshow('gray', thres)
    # cv2.imshow('edges', edges)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    # r = cv2.getTrackbarPos('rho', 'image')
    # t = cv2.getTrackbarPos('theta', 'image')
    # tr = cv2.getTrackbarPos('threshold', 'image')
    # lines = cv2.HoughLines(edges, 3, np.pi / abs(90), 135)
    # if not ((lines is None) or (len(lines) == 0)):
    #     for r, t in lines[0]:
    #         a = np.cos(t)
    #         b = np.sin(t)
    #         x0 = a * r
    #         y0 = b * r
    #         x1 = int(x0 + 1000 * (-b))
    #         y1 = int(y0 + 1000 * a)
    #         x2 = int(x0 - 1000 * (-b))
    #         y2 = int(y0 - 1000 * a)
    #         cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1)

    lines = cv2.HoughLines(edges, 3, abs(np.pi / 90), 135)
    # lines = [[lines[0][1], lines[0][5]]]
    #unique_lines = lines

    #unique_lines = find_unique_lines(lines)
    unique_lines = [[lines[0][0], lines[0][7]]]
    unique_lines = find_unique_lines(unique_lines)

    if not ((unique_lines is None) or (len(unique_lines) == 0)):
        cnt = 0
        for r, t in unique_lines:
            a = np.cos(t)
            b = np.sin(t)
            x0 = a * r
            y0 = b * r
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)
            cv2.line(img, (x1, y1), (x2, y2), (cnt, 0, 255 - cnt), 2)
            # cv2.putText(img, str(cnt / 20), (int(x0) + 5, int(y0) + 30), cv2.FONT_HERSHEY_PLAIN, fontScale=2,
            #             color=(cnt, 0, 255 - cnt), thickness=2)
            # cv2.putText(img, "{:.1f}".format(x0) + '; ' + "{:.1f}".format(y0), (int(x0), int(y0) + 50),
            #             cv2.FONT_HERSHEY_PLAIN,
            #             fontScale=1, color=(cnt, 0, 255 - cnt), thickness=2)
            # cnt += 20

    # inter_points = find_intersections(unique_lines)
    # # result_points = find_unique_points(inter_points)
    # # for x, y in inter_points:
    # #     cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), thickness=3)
    #
    # unique_abc_lines = transform_lines(unique_lines)
    # test_lines = unique_abc_lines
    # test_edges = extract_vertices_and_edges(inter_points, test_lines)
    #
    # cnt = 0
    # for a, b, c in test_lines:
    #     if b == 0:
    #         x = -c / a
    #         cv2.line(img, (int(x), 0), (int(x), 1000), (0, 0, 255), 1)
    #     else:
    #         x1 = 0
    #         y1 = -c / b
    #         x2 = 1000
    #         y2 = (-a * x2 - c) / b
    #         cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
    #     cnt += 20
    #
    # cnt = 0
    # for p1, p2 in test_edges:
    #     cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, cnt*2, cnt), thickness=3)
    #     cnt += 20