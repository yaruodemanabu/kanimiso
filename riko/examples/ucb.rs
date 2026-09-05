use riko::Ucb;
fn main() -> riko::Result<()> {
    let rewards = [[0.1, 0.8], [0.2, 0.7], [0.0, 0.9]];
    let mut policy = Ucb::new(2, 2.0_f64.sqrt())?;
    for outcomes in rewards {
        let choice = policy.select();
        policy.update(choice, outcomes[choice.arm()])?;
    }
    println!("pulls: {:?}", policy.pulls());
    Ok(())
}
