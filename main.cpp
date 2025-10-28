#include <matplot/matplot.h>

#include <algorithm>
#include <format>
#include <iostream>
#include <random>
#include <ranges>

using namespace matplot;

// Problem parameters
int x = 10;
int y = 10;

int goalx = 8;
int goaly = 6;

int startx = 0;
int starty = 4;

// Model parameters
int steps = 10000;
int discount = 1;
double epsilon = 0.1;
double alpha = 0.5;
int num_simulations = 1000;

// Engine

enum class Action : uint8_t { Left,
                              Up,
                              Right,
                              Down };

constexpr std::array<std::pair<int, int>, 4> delta{{
    {-1, 0},
    {0, 1},
    {1, 0},
    {0, -1},
}};

Action eps_choose_action(const std::vector<double>& candidates, std::bernoulli_distribution& greedyDist,
                         std::uniform_int_distribution<>& uniform_dist, std::mt19937& gen) {
    bool choseGreedy = greedyDist(gen);
    if (choseGreedy) {
        return static_cast<Action>(std::ranges::distance(candidates.begin(), std::ranges::max_element(candidates)));
    } else {
        return static_cast<Action>(uniform_dist(gen));
    }
}

double take_action(int& curx, int& cury, Action& a, const std::vector<int>& wind) {
    cury += delta[static_cast<size_t>(a)].second + wind[curx];
    curx += delta[static_cast<size_t>(a)].first;
    if (curx >= x) {
        curx = x - 1;
    }
    if (curx < 0) {
        curx = 0;
    }
    if (cury >= y) {
        cury = y - 1;
    }
    if (cury < 0) {
        cury = 0;
    }
    if (curx == goalx && cury == goaly) {
        return 0;
    } else
        return -1;
}

// For graphing
void init_plot() {
    auto be = figure(false)->backend();
    be->run_command("unset warnings");
    ylim({0, 100});
    hold(on);
}
void init_grid() {
    gca()->children({});
    ylabel("");
    auto ax = gca();
    ax->xticks({});
    ax->yticks({});
    axis({0.0f, (float)x, -1.0f, (float)y});
    xlabel("Wind");
    title("Playout based on Policy (10000 steps)");
}
static void draw_grid(int cols, int rows,
                      double lw = 2.5, float g = 0.7) {
    for (int c = 0; c < cols; ++c) {
        auto v = plot(std::vector<double>{(double)c, (double)c},
                      std::vector<double>{0.0, (double)rows});
        v->color({g, g, g}).line_width(lw);
    }
    for (int r = 0; r < rows; ++r) {
        auto h = plot(std::vector<double>{0.0, (double)cols},
                      std::vector<double>{(double)r, (double)r});
        h->color({g, g, g}).line_width(lw);
    }
}

static void draw_wind(int cols, int rows, const std::vector<int>& wind) {
    for (int i = 0; i < cols; i++) {
        text(i + 0.37, -0.55, std::format("{}", wind[i]));
    }
}

std::vector<double> average_runs(const std::vector<std::vector<int>>& episode_lengths_s) {
    std::size_t maxT = 0;
    for (auto& v : episode_lengths_s)
        if (v.size() > maxT) maxT = v.size();

    std::vector<double> sum(maxT, 0.0);
    std::vector<std::size_t> cnt(maxT, 0);

    for (auto& v : episode_lengths_s) {
        for (std::size_t t = 0; t < v.size(); ++t) {
            sum[t] += v[t];
            ++cnt[t];
        }
    }

    std::vector<double> avg(maxT, 0.0);
    for (std::size_t t = 0; t < maxT; ++t) {
        avg[t] = cnt[t] ? (sum[t] / static_cast<double>(cnt[t])) : 0.0;
    }
    return avg;
}

int main(int argc, char* argv[]) {  // set seed if you want
    unsigned int seed;
    if (argc > 1) {
        seed = static_cast<unsigned int>(std::stoul(argv[1]));
    } else {
        seed = std::random_device{}();
    }
    std::mt19937 gen{seed};
    std::normal_distribution<> normal_dist(1.0, 1.0);
    std::discrete_distribution<> dist({15, 20, 30, 20, 15}); // wind speed has 15% chance of +-2, 20% chance of +-1, 30% chance of 0
    std::bernoulli_distribution greedyDist(1.0 - epsilon);
    std::uniform_int_distribution<> uniform_int_dist(0, 4);
    std::vector<std::vector<int>> episode_lengths_s(num_simulations);
    std::vector<std::vector<int>> episode_lengths_q(num_simulations);

    // On-Policy SARSA
    auto run_SARSA = [&](size_t i, const std::vector<int>& wind) {
        std::vector Q_s(x, std::vector(y, std::vector<double>(4)));  // SARSA state action value
        int curx = startx, cury = starty, prevt = 0;
        Action chosen_action, next_action;
        for (int t = 0; t < steps; t++) {
            if (curx == startx && cury == starty) {
                chosen_action = eps_choose_action(Q_s[curx][cury], greedyDist, uniform_int_dist, gen);
            }
            int prevx = curx, prevy = cury;
            double reward = take_action(curx, cury, chosen_action, wind);
            next_action = eps_choose_action(Q_s[curx][cury], greedyDist, uniform_int_dist, gen);
            Q_s[prevx][prevy][static_cast<size_t>(chosen_action)] += alpha * (reward + discount * Q_s[curx][cury][static_cast<size_t>(next_action)] -
                                                                              Q_s[prevx][prevy][static_cast<size_t>(chosen_action)]);
            chosen_action = next_action;
            if (curx == goalx && cury == goaly) {
                episode_lengths_s[i].push_back(t - prevt);
                prevt = t;
                curx = startx;
                cury = starty;
            }
        }
        return Q_s;
    };
    // Q-learning
    auto run_Q_learning = [&](size_t i, const std::vector<int>& wind) {
        std::vector Q_q(x, std::vector(y, std::vector<double>(4)));  // Q-learning state action value
        int curx = startx, cury = starty, prevt = 0;
        Action chosen_action;
        for (int t = 0; t < steps; t++) {
            chosen_action = eps_choose_action(Q_q[curx][cury], greedyDist, uniform_int_dist, gen);
            int prevx = curx, prevy = cury;
            double reward = take_action(curx, cury, chosen_action, wind);
            Q_q[prevx][prevy][static_cast<size_t>(chosen_action)] += alpha * (reward + discount * *std::ranges::max_element(Q_q[curx][cury]) -
                                                                              Q_q[prevx][prevy][static_cast<size_t>(chosen_action)]);
            if (curx == goalx && cury == goaly) {
                episode_lengths_q[i].push_back(t - prevt);
                prevt = t;
                curx = startx;
                cury = starty;
            }
        }
        return Q_q;
    };

    // Could have Parallelized the simulations:
    // auto idx = std::views::iota(0, num_simulations);
    // std::for_each(std::execution::par, idx.begin(), idx.end(),
    //           run_SARSA);
    // std::for_each(std::execution::par, idx.begin(), idx.end(),
    //           run_Q_learning);
    for (size_t i = 0; i < num_simulations; i++) {
        std::vector<int> wind(x);
        std::ranges::generate(wind, [&]() { return dist(gen) - 2; });
        run_SARSA(i, wind);
        run_Q_learning(i, wind);
    }
    std::vector<double> averaged_episode_lengths_s = average_runs(episode_lengths_s);
    std::vector<double> averaged_episode_lengths_q = average_runs(episode_lengths_q);

    std::vector<int> episodes_s(averaged_episode_lengths_s.size());
    std::vector<int> episodes_q(averaged_episode_lengths_q.size());
    std::iota(episodes_s.begin(), episodes_s.end(), 1);
    std::iota(episodes_q.begin(), episodes_q.end(), 1);
    init_plot();
    plot(episodes_s, averaged_episode_lengths_s)->line_width(2).color("blue").display_name("SARSA");
    plot(episodes_q, averaged_episode_lengths_q)->line_width(2).color("red").display_name("Q-Learning");
    title("Averaged over 1000 runs with random wind speeds (1 run = 10000 steps)");
    ylabel("Episode Length (no. of steps)");
    xlabel("Episode");
    auto legend = ::matplot::legend({});
    show();
    save("graph.png");

    // Run one SARSA and Q Learning again to plot policy
    {
        std::vector<int> wind(x);
        std::ranges::generate(wind, [&]() { return dist(gen) - 2; });

        auto Q_s_plot = run_SARSA(1, wind);
        auto Q_q_plot = run_Q_learning(1, wind);
        init_grid();
        plot({0})->color("blue").line_width(3);
        plot({0})->color("red").line_width(3);  // For legend to display correct colour
        draw_grid(x, y);
        draw_wind(x, y, wind);
        auto plot_policy = [&](std::vector<std::vector<std::vector<double>>> Q, std::string color) {
            int curx = startx, cury = starty;
            std::vector<std::pair<int, int>> visited;
            auto already_visited = [&](int x, int y) {
                return std::find(visited.begin(), visited.end(), std::pair{x, y}) != visited.end();
            };
            while (!(curx == goalx && cury == goaly) && !already_visited(curx, cury)) {
                visited.push_back({curx, cury});
                auto zero_epsilon = std::bernoulli_distribution{1.0};
                Action a = eps_choose_action(Q[curx][cury], zero_epsilon, uniform_int_dist, gen);
                take_action(curx, cury, a, wind);
            }
            if (curx != goalx || cury != goaly) {
                visited.push_back({curx, cury});
            } else {
                visited.push_back({goalx, goaly});
            }

            for (size_t i = 0; i < visited.size() - 1; i++) {
                auto a = arrow(visited[i].first + 0.5, visited[i].second + 0.5, visited[i + 1].first + 0.5, visited[i + 1].second + 0.5)->color(color);
                if (i == 0) {
                    a.display_name("Q-Learning");
                }
            }
        };
        plot_policy(Q_s_plot, "blue");
        plot_policy(Q_q_plot, "red");
        auto lgd = ::matplot::legend({"SARSA", "Q-learning"});
        show();
        save("graph2.png");
    }
}