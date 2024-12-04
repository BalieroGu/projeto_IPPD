#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

typedef struct {
    float x;
    float y;
} Point;

typedef struct {
    char label;
    int length;
    Point *points;
} Group;

void on_error() {
    printf("Invalid input file.\n");
    exit(1);
}

int parse_number_of_groups() {
    int n;
    if (scanf(" n_groups=%d ", &n) != 1) on_error();
    return n;
}

Point parse_point() {
    float x, y;
    if (scanf(" (%f ,%f) ", &x, &y) != 2) on_error();
    Point point;
    point.x = x;
    point.y = y;
    return point;
}

Group parse_next_group() {
    char label;
    int length;

    if (scanf(" label=%c ", &label) != 1) on_error();
    if (scanf(" length=%d ", &length) != 1) on_error();

    Group group;
    group.label = label;
    group.length = length;
    group.points = (Point *)malloc(sizeof(Point) * length);

    for (int i = 0; i < length; i++) {
        group.points[i] = parse_point();
    }

    return group;
}

int parse_k() {
    int k;
    if (scanf(" k=%d ", &k) != 1) on_error();
    return k;
}

float euclidean_distance_no_sqrt(Point a, Point b) {
    return ((b.x - a.x) * (b.x - a.x)) + ((b.y - a.y) * (b.y - a.y));
}

int compare_for_sort(const void *a, const void *b) {
    return *(char *)a - *(char *)b;
}

char knn(int n_groups, Group *groups, int k, Point to_evaluate) {
    char *labels = (char *)malloc(sizeof(char) * k);
    float *distances = (float *)malloc(sizeof(float) * k);

    for (int i = 0; i < k; i++) {
        labels[i] = -1;
        distances[i] = -1;
    }

    // Paralelização do cálculo de distâncias
    #pragma omp parallel
    {
        char *local_labels = (char *)malloc(sizeof(char) * k);
        float *local_distances = (float *)malloc(sizeof(float) * k);

        for (int i = 0; i < k; i++) {
            local_labels[i] = -1;
            local_distances[i] = -1;
        }

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < n_groups; i++) {
            Group g = groups[i];
            for (int j = 0; j < g.length; j++) {
                float d = euclidean_distance_no_sqrt(to_evaluate, g.points[j]);

                for (int x = 0; x < k; x++) {
                    if (d < local_distances[x] || local_distances[x] == -1) {
                        for (int y = k - 1; y > x; y--) {
                            local_distances[y] = local_distances[y - 1];
                            local_labels[y] = local_labels[y - 1];
                        }
                        local_distances[x] = d;
                        local_labels[x] = g.label;
                        break;
                    }
                }
            }
        }

        // Combina os resultados locais na thread principal
        #pragma omp critical
        {
            for (int i = 0; i < k; i++) {
                if (local_distances[i] < distances[k - 1] || distances[k - 1] == -1) {
                    for (int x = k - 1; x > 0; x--) {
                        distances[x] = distances[x - 1];
                        labels[x] = labels[x - 1];
                    }
                    distances[0] = local_distances[i];
                    labels[0] = local_labels[i];
                }
            }
        }

        free(local_labels);
        free(local_distances);
    }

    qsort(labels, k, sizeof(char), compare_for_sort);

    char most_frequent = labels[0];
    int most_frequent_count = 1;
    int current_frequency = 1;

    for (int i = 1; i < k; i++) {
        if (labels[i] != labels[i - 1]) {
            if (current_frequency > most_frequent_count) {
                most_frequent = labels[i - 1];
                most_frequent_count = current_frequency;
            }
            current_frequency = 1;
        } else {
            current_frequency++;
        }

        if (i == k - 1 && current_frequency > most_frequent_count) {
            most_frequent = labels[i - 1];
            most_frequent_count = current_frequency;
        }
    }

    free(labels);
    free(distances);
    return most_frequent;
}

int main() {
    int n_groups = parse_number_of_groups();
    Group *groups = (Group *)malloc(sizeof(Group) * n_groups);

    for (int i = 0; i < n_groups; i++) {
        groups[i] = parse_next_group();
    }

    int k = parse_k();
    Point to_evaluate = parse_point();

    printf("%c\n", knn(n_groups, groups, k, to_evaluate));

    for (int i = 0; i < n_groups; i++) {
        free(groups[i].points);
    }
    free(groups);

    return 0;
}