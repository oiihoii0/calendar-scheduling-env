"""Generate a sample Gantt chart for the README."""
import matplotlib
matplotlib.use("Agg")

from scheduling_env import CalendarEnv
from scheduling_env.baseline import HeuristicAgent
from scheduling_env.visualize import render_gantt

env = CalendarEnv("constrained_scheduling")
obs, info = env.reset(seed=42)
agent = HeuristicAgent()

done = False
while not done:
    action = agent.act(obs, info, env)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.render()
render_gantt(env, save_path="schedule_gantt.png", show=False)
print("Gantt chart saved to schedule_gantt.png")
