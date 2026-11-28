"""Authors: Team 8; assisted by OpenAI Codex. External sources: none.
Development tests only; not part of the submitted experiment workflow.
Run: python -m unittest discover -s src/tools/v6 -p test_reporting.py
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import benchmark as bench


def report(runs=5):
    sample = dict(gray=1., blur=1., harris=1., nms_topk=1., describe=1., match=1.,
                  ransac=1., warp_blend=1., kernel_total=8., h2d=1., d2h=1., end_to_end=12.)
    return dict(status='ok', runs=runs, warmup_runs=1, timings_ms=sample,
                stage_timings_ms=dict(S1=2., S2=2., S3=1., S4=1., S5=1., S6=1.),
                samples_ms=[dict(sample) for _ in range(runs)])


def write_pair(directory, case, first=5):
    start=[first,11-first]
    order=[v for i in range(5) for v in (start if i%2==0 else start[::-1])]
    manifest=dict(status='ok',measurement_mode='same_process_interleaved',
                  runs_per_version=5,warmup_runs_per_version=1,order=order)
    for version in (5,6):
        data=report(5)
        data.update(gpu_name='test GPU',measurement_mode='same_process_interleaved',
                    sample_positions=[i for i,v in enumerate(order) if v==version])
        bench.write_json(directory/f'{case}_v{version}.json',data)
    bench.write_json(directory/f'{case}_paired.json',manifest)
    return manifest


def fake_invoke(command, log, output):
    if command[1]=='--compare':
        return write_pair(Path(command[4]),command[5],command[7])
    data = dict(status='pass',checks={'matches':True}) if 'verify' in str(command[0]) else report(int(command[5]))
    bench.write_json(output,data)
    return data


class ReportingTests(unittest.TestCase):
    def test_sample_mean_must_match(self):
        data=report(); data['samples_ms'][0]['match']=10.
        with self.assertRaises(ValueError): bench.timing_fields({},6,data)

    def test_reject_insufficient_or_nonfinite_samples(self):
        for invalid in ('missing','nan'):
            data=report()
            if invalid=='missing': data['samples_ms'].pop()
            else: data['samples_ms'][0]['match']=float('nan')
            with self.assertRaises(ValueError): bench.timing_fields({},6,data)

    def test_minimum_saved_for_stages(self):
        row={}; bench.timing_fields(row,6,report())
        self.assertEqual(row['v6_S1_min_ms'],2.)
        self.assertEqual(row['v6_match_min_ms'],1.)
        self.assertNotIn('v6_match_cv',row)

    def test_failed_cases_have_no_successful_speedup(self):
        with tempfile.TemporaryDirectory() as root, patch.object(bench,'cases',return_value=[('case',Path('a'),Path('b'))]):
            directory=Path(root)
            bench.write_json(directory/'case_v0.json',report(5))
            write_pair(directory,'case')
            for status in ('correctness_failed','consistent_pipeline_failure','error','pass'):
                bench.write_json(directory/'case_verification.json',dict(status=status,checks={'matches':status=='pass'}))
                row=bench.summarize(directory,'optimized')[0]
                self.assertEqual(row['status'],status)
                self.assertEqual('v5_over_v6_match_speedup' in row,status=='pass')

    def test_baseline_and_optimized_use_versions_0_5_6(self):
        environment={'gpu':{'text':'test GPU'},'session':'test-session'}
        commands=[]
        def record_invoke(command, log, output):
            commands.append(command)
            return fake_invoke(command,log,output)
        with tempfile.TemporaryDirectory() as root:
            directory=Path(root)/'run'
            with patch.object(bench,'cases',return_value=[('case',Path('a'),Path('b'))]), \
                 patch.object(bench,'environment',return_value=environment), \
                 patch.object(bench,'invoke',side_effect=record_invoke):
                for phase in ('baseline','optimized'):
                    with patch('sys.argv',['benchmark.py','--phase',phase,'--run-dir',str(directory)]):
                        bench.main()
            versions=[command[4] for command in commands if 'benchmark' in str(command[0]) and command[1]!='--compare']
            self.assertEqual(versions,[0,5,0])
            paired=[command for command in commands if command[1]=='--compare']
            self.assertEqual(len(paired),1)
            self.assertEqual(paired[0][6],5)
            self.assertFalse((directory/'case_v6.json').exists())
            self.assertEqual(bench.load_report(directory/'summary.json')[0]['status'],'baseline_only')
            directory=directory/'comparison'
            row=bench.load_report(directory/'summary.json')[0]
            self.assertEqual(row['status'],'pass')
            self.assertEqual(row['v5_over_v6_match_speedup'],1.)

    def test_comparison_alternates_gpu_order_and_keeps_baseline(self):
        environment={'gpu':{'text':'test GPU'},'session':'test-session'}
        with tempfile.TemporaryDirectory() as root:
            directory=Path(root)/'run'
            with patch.object(bench,'cases',return_value=[('a',Path('a'),Path('b')),('b',Path('a'),Path('b'))]), \
                 patch.object(bench,'environment',return_value=environment), \
                 patch.object(bench,'invoke',side_effect=fake_invoke):
                for phase in ('baseline','optimized'):
                    with patch('sys.argv',['benchmark.py','--phase',phase,'--run-dir',str(directory)]): bench.main()
            comparison=directory/'comparison'
            self.assertEqual(bench.load_report(comparison/'a_paired.json')['order'],[5,6,6,5,5,6,6,5,5,6])
            self.assertEqual(bench.load_report(comparison/'b_paired.json')['order'],[6,5,5,6,6,5,5,6,6,5])
            self.assertTrue(bench.validate_pair(comparison,'a')[0])
            self.assertTrue(bench.validate_pair(comparison,'b')[0])
            self.assertTrue((directory/'a_v5.json').is_file())
            self.assertTrue((comparison/'a_v5.json').is_file())
            self.assertTrue((comparison/'a_v0.json').is_file())

    def test_reject_unbalanced_or_mismatched_paired_samples(self):
        with tempfile.TemporaryDirectory() as root:
            directory=Path(root)
            for fault in ('order','positions','samples','mode','gpu'):
                write_pair(directory,'case')
                if fault=='order':
                    data=bench.load_report(directory/'case_paired.json'); data['order']=[5,6]*5
                    bench.write_json(directory/'case_paired.json',data)
                else:
                    data=bench.load_report(directory/'case_v6.json')
                    if fault=='positions': data['sample_positions']=[1]*5
                    if fault=='samples': data['samples_ms'].pop()
                    if fault=='mode': data['measurement_mode']='single_version'
                    if fault=='gpu': data['gpu_name']='other GPU'
                    bench.write_json(directory/'case_v6.json',data)
                self.assertFalse(bench.validate_pair(directory,'case')[0],fault)

    def test_paired_process_failure_invalidates_both_reports(self):
        with tempfile.TemporaryDirectory() as root:
            directory=Path(root)
            write_pair(directory,'case')
            with patch.object(bench,'invoke',return_value={'status':'process_error','error':'child failed'}):
                bench.invoke_pair(Path('a'),Path('b'),directory,'case',5)
            for version in (5,6):
                self.assertEqual(bench.load_report(directory/f'case_v{version}.json')['status'],'process_error')

    def test_reject_baseline_from_another_session(self):
        with tempfile.TemporaryDirectory() as root:
            directory=Path(root)
            bench.write_json(directory/'environment.json',{'gpu':{'text':'same GPU'},'session':'old'})
            with patch.object(bench,'environment',return_value={'gpu':{'text':'same GPU'},'session':'new'}), \
                 patch('sys.argv',['benchmark.py','--phase','optimized','--run-dir',str(directory)]):
                with self.assertRaisesRegex(RuntimeError,'GPU identity changed'):
                    bench.main()

    def test_profiler_selects_same_explicit_case_for_both_versions(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('v6_profile_test',Path(__file__).with_name('profile.py'))
        profiler=importlib.util.module_from_spec(spec); spec.loader.exec_module(profiler)
        case='location1_large_4k'
        with tempfile.TemporaryDirectory() as root:
            directory=Path(root)
            bench.write_json(directory/f'{case}_v5.json',report())
            other=report(); other['timings_ms']['match']=100.
            bench.write_json(directory/'location2_medium_1080p_v5.json',other)
            with patch.object(profiler,'environment',return_value={}), patch.object(profiler.shutil,'which',return_value=None):
                for version in (5,6):
                    with patch('sys.argv',['profile.py','--run-dir',str(directory),'--version',str(version),'--case',case]):
                        profiler.main()
                    manifest=bench.load_report(directory/'profiling'/case/f'profile_v{version}.json')
                    self.assertEqual(manifest['case'],case)
                    self.assertEqual(manifest['version'],version)
                    self.assertEqual(manifest['status'],'unavailable')


if __name__=='__main__': unittest.main()
