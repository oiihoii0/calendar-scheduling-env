from scheduling_env import CalendarEnv

# Pick a difficulty: "simple_scheduling", "constrained_scheduling", or "complex_scheduling"
env = CalendarEnv(task_name="simple_scheduling")

obs, info = env.reset(seed=42)
print("Task:", info["description"])

done = False
while not done:
    action = env.action_space.sample()  # replace with your agent's logic
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.render()           # pretty-print the schedule
print(env.get_state()) # full state snapshot (OpenEnv spec)
